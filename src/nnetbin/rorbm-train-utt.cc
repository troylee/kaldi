// nnetbin/rorbm-train-utt.cc
/*
 * Troy Lee
 *
 */

#include "nnet/nnet-rorbm.h"
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
inline std::string to_string(const T& t)
                             {
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
    const char *usage =
        "Perform per utterance RoRbm training by contrastive divergence alg.\n"
            "Usage:  rorbm-train-utt [options] <model-in> <feature-rspecifier> <model-out> [<epoch-weight>]\n"
            "e.g.: \n"
            " rorbm-train-utt rorbm.init scp:train.scp rorbm.final rorbm.epoch\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    int32 cachesize = 32768,
        bunchsize = 512,
        maxepoch = 1000,
        num_gibbs = 100,
        momentum_change_epoch = 5,
        numInferIters = 50;

    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("maxepoch", &maxepoch, "Maximum number of epochs for training");
    po.Register("num-gibbs", &num_gibbs, "Number of CD iterations");
    po.Register("momentum-change-epoch", &momentum_change_epoch,
                "The epoch iteration to change momentum");

    BaseFloat init_momentum = 0.5,
        high_momentum = 0.9;

    po.Register("init-momentum", &init_momentum, "Initial training momentum");
    po.Register("high-momentum", &high_momentum, "Higher training momentum");

    std::string feature_transform = "";

    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        target_model_filename = po.GetArg(3),
        epoch_model_filename = po.GetOptArg(4);

    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility

    Nnet rorbm_transf;
    if (feature_transform != "") {
      rorbm_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRoRbm);
    RoRbm &rorbm = dynamic_cast<RoRbm&>(*nnet.Layer(0));

    //TODO:: general configurations such as learn rates

    rorbm.SetNumInferenceIters(numInferIters);

    CuMatrix<BaseFloat> feats, feats_transf, vt, vt_recon;
    CuMatrix<BaseFloat> dummy_mse_mat;
    CuVector<BaseFloat> s_mu;
    std::vector<int32> dummy_cache_vec;
    std::string epoch_name;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    int32 num_done = 0; // num_cache = 0;

    while (!feature_reader.Done()) {
      // get feature matrix
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      std::string key = feature_reader.Key();

      // push features to GPU
      feats.CopyFromMat(mat);
      // possibly apply transform
      rorbm_transf.Feedforward(feats, &feats_transf);

      rorbm.SetMomentum(init_momentum);

      /* initialize the moving average of the mean of the mask */
      s_mu.Resize(rorbm.VisDim());
      s_mu.Set(0.9);

      Timer tot_tim;
      KALDI_LOG<< "##################################################################";
      KALDI_LOG<< "UTT: " << key;
      KALDI_LOG<< "RoRBM TRAINING STARTED [" << currentDateTime() << "]";

      rorbm.InitTraining(feats.NumRows());

      for (int32 epoch = 0; epoch < maxepoch; ++epoch) {

        Timer tim;
        double time_next = 0;
        KALDI_LOG<< "******************************************************************";
        KALDI_LOG<< "Epoch " << epoch << " started [" << currentDateTime() << "]";

        kaldi::int64 tot_t = 0;

        MseProgress mse;

        // change momentum if ready
        if (epoch == momentum_change_epoch) {
          rorbm.SetMomentum(high_momentum);
        }

        // TRAIN with CD
        if (epoch == 0) {
          rorbm.InitParticle(feats_transf);
        }

        /* positive phase */
        // forward pass, compute ha, hs, s and v_condmean given the vt_cn
        rorbm.Inference(feats_transf);
        rorbm.GetReconstruction(&vt_recon);
        // compute positive gradient
        rorbm.CollectPositiveStats(feats_transf, &s_mu);

        /* negative phase */
        // Gibbs sampling, SAP
        for (int32 iter = 0; iter < num_gibbs; ++iter) {
          // Using fantasy particles to do the learning
          rorbm.SAPIteration();
        }
        // compute negative gradient
        rorbm.CollectNegativeStats(s_mu);

        // update step
        rorbm.RoRbmUpdate();

        // evaluate mean square error
        mse.Eval(feats_transf, vt_recon, &dummy_mse_mat);

        tot_t += vt.NumRows();

        // next feature file...
        Timer t_features;
        feature_reader.Next();
        time_next += t_features.Elapsed();

        std::cout << "\n" << std::flush;

        KALDI_LOG<< "Epoch " << epoch << " finished [" << currentDateTime() << "] "
        << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
        << ", feature wait " << time_next << "s";

        KALDI_LOG<< mse.Report();
      }

        // write the final model to output file
      nnet.Write(target_model_filename + "." + key, binary);

    }

    KALDI_LOG<< "Done " << num_done << " files.";

    KALDI_LOG<< "******************************************************************";
    KALDI_LOG<< "Model saved to " << target_model_filename;
    KALDI_LOG<< "RoRBM TRAININIG FINISHED [" << currentDateTime() << "]";
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
