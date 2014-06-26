// nnetbin/grbm-train-frmshuff.cc
/*
 * Troy Lee
 *
 */

#include "nnet/nnet-grbm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"

#include <time.h>
#include <sstream>

namespace kaldi {

template<class T>
inline std::string to_string(const T& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
  // for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

  return buf;
}

}  // end namespace

int main(int argc, char *argv[]) {

  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {

    const char *usage = "Perform GRBM training by contrastive divergence alg.\n"
        "Usage:  grbm-train-frmshuff [options] <model-in> <feature-rspecifier> "
        "<model-out> [<epoch-weight>]\n"
        "e.g.: \n"
        " grbm-train-frmshuff rbm.init scp:train.scp rbm.final rbm.epoch\n";

    /*
     * With numCD=1, apply_sparsity=false, enable_vis_random=false, this is the same as
     */

    ParseOptions po(usage);
    bool binary = false, apply_sparsity = true, enable_vis_random = true,
        enable_hid_random = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("apply-sparsity", &apply_sparsity,
                "Whether to use sparsity in the hidden activations");
    po.Register("enable-visible-randomness", &enable_vis_random,
                "Add randomness to the visible reconstructions");
    po.Register("enable-hidden-randomness", &enable_hid_random,
                "Add randomness to the hidden reconstructions");

    BaseFloat learn_rate = 0.001, init_momentum = 0.5, high_momentum = 0.9,
        l2_penalty = 0.0002, var_learn_rate = 0.001, sparsity_lambda = 0.01,
        sparsity_p = 0.2;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("init-momentum", &init_momentum, "Initial momentum");
    po.Register("high-momentum", &high_momentum, "Higher momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("var-learn-rate", &var_learn_rate,
                "Input variance learning rate");
    po.Register("sparsity-lambda", &sparsity_lambda,
                "Lambda for weight sparsity");
    po.Register("sparsity-p", &sparsity_p, "Parameter p for weight sparsity");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    int32 bunchsize = 512, cachesize = 32768, maxEpoch = 1000, numCD = 100,
        momentum_change_epoch = 5, var_start_iter = 1;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");
    po.Register("maxEpoch", &maxEpoch, "Maximum number of epochs for training");
    po.Register("numCD", &numCD, "Number of CD iterations");
    po.Register("momentum-change-epoch", &momentum_change_epoch,
                "Epoch to use high momentum");
    po.Register("var-start-epoch", &var_start_iter,
                "The iteration to start learning visible variance");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1), feature_rspecifier = po.GetArg(
        2);

    std::string target_model_filename;
    target_model_filename = po.GetArg(3);

    std::string epoch_model_filename;
    epoch_model_filename = po.GetOptArg(4);

    CuRand<BaseFloat> cu_rand;

    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility

    Nnet grbm_transf;
    if (feature_transform != "") {
      grbm_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kGRbm);
    GRbm &grbm = dynamic_cast<GRbm&>(*nnet.Layer(0));

    grbm.SetLearnRate(learn_rate);
    grbm.SetMomentum(init_momentum);  // initial momentum
    grbm.SetL2Penalty(l2_penalty);  // weight cost
    grbm.SetVarianceLearnRate(0.0);  // initial variance leanring rate

    if (apply_sparsity) {
      grbm.EnableSparsity();
      grbm.ConfigSparsity(sparsity_lambda, sparsity_p);
    } else {
      grbm.DisableSparsity();
    }

    CuMatrix<BaseFloat> feats, feats_transf, pos_vis, pos_hid, pos_hid_states,
        neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;
    CuVector<BaseFloat> avg_hid_probs;
    std::vector<int32> dummy_cache_vec;
    std::string epoch_name;

    Timer tot_tim;
    KALDI_LOG<< "##################################################################";
    KALDI_LOG<< "GRBM TRAINING STARTED [" << currentDateTime() << "]";

    bool first_bunch = true;  // indicating the first bunch of data for the whole training process
    for (int32 epoch = 0; epoch < maxEpoch; ++epoch) {

      Timer tim;
      double time_next = 0;
      KALDI_LOG<< "******************************************************************";
      KALDI_LOG<< "Epoch " << epoch << " started [" << currentDateTime() << "]";

      kaldi::int64 tot_t = 0;

      MseProgress mse;

      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      Cache cache;
      cache.Init(cachesize, bunchsize);

      // configure the variance learning rate
      if (epoch == var_start_iter) {
        grbm.SetVarianceLearnRate(var_learn_rate);
      }

      // change momentum if ready
      if (epoch == momentum_change_epoch) {
        grbm.SetMomentum(high_momentum);
      }

      int32 num_done = 0, num_cache = 0;
      while (1) {
        // fill the cache
        while (!cache.Full() && !feature_reader.Done()) {
          // get feature matrix
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          // resize the dummy vector to fill Cache:: with
          dummy_cache_vec.resize(mat.NumRows());
          // push features to GPU
          feats.CopyFromMat(mat);
          // possibly apply transform
          grbm_transf.Feedforward(feats, &feats_transf);
          // add to cache
          cache.AddData(feats_transf, dummy_cache_vec);
          num_done++;
          // next feature file...
          Timer t_features;
          feature_reader.Next();
          time_next += t_features.Elapsed();
        }
        // randomize
        cache.Randomize();
        // report
        std::cerr << "Cache #" << ++num_cache << " "
            << (cache.Randomized() ? "[RND]" : "[NO-RND]") << " segments: "
            << num_done << " frames: " << tot_t << "\n";
        // train with the cache
        while (!cache.Empty()) {
          // get block of feature/target pairs
          cache.GetBunch(&pos_vis, &dummy_cache_vec);

          // TRAIN with CD
          /* positive phase */
          // forward pass
          grbm.Propagate(pos_vis, &pos_hid);
          if (enable_hid_random) {
            // generate binary hidden states
            cu_rand.BinarizeProbs(pos_hid, &pos_hid_states);
          } else {
            // just copy the probabilities
            pos_hid_states.CopyFromMat(pos_hid);
          }

          /* negative phase */
          // CD
          for (int32 iterCD = 0; iterCD < numCD; ++iterCD) {
            // reconstruct visible
            grbm.Reconstruct(pos_hid_states, &neg_vis);
            if (enable_vis_random) {
              // add Gaussian noise
              grbm.SampleVisible(cu_rand, &neg_vis);
            }
            // forward to generate hidden probabilities
            grbm.Propagate(neg_vis, &neg_hid);
            if (enable_hid_random) {
              // generate binary hidden states
              cu_rand.BinarizeProbs(neg_hid, &pos_hid_states);
            } else {
              // just copy the probabilities
              pos_hid_states.CopyFromMat(neg_hid);
            }
          }

          // update step
          grbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid, &avg_hid_probs,
                         first_bunch);
          if (first_bunch)
            first_bunch = false;
          // evaluate mean square error
          mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

          tot_t += pos_vis.NumRows();
        }

        // stop training when no more data
        if (feature_reader.Done())
          break;
      }

      if (epoch_model_filename != "") {
        nnet.Write(epoch_model_filename + "_epoch" + to_string(epoch), binary);
      }

      std::cout << "\n" << std::flush;

      KALDI_LOG<< "Epoch " << epoch << " finished [" << currentDateTime() << "] "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
      << ", feature wait " << time_next << "s";

      KALDI_LOG<< "Done " << num_done << " files.";

      KALDI_LOG<< mse.Report();

    }

      // write the final model to output file
    nnet.Write(target_model_filename, binary);

    KALDI_LOG<< "******************************************************************";
    KALDI_LOG<< "Model saved to " << target_model_filename;
    KALDI_LOG<< "GRBM TRAININIG FINISHED [" << currentDateTime() << "]";
    KALDI_LOG<< "##################################################################";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
