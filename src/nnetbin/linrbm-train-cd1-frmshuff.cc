// nnetbin/linrbm-train-cd1-frmshuff.cc

#include "nnet/nnet-linrbm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of LinRbm training by contrastive divergence alg.\n"
            "Usage:  linrbm-train-cd1-frmshuff [options] <model-in> <feature-rspecifier> [<model-out>]\n"
            "e.g.: \n"
            " linrbm-train-cd1-frmshuff rbm.init scp:train.scp rbm.iter1\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    bool cross_validate = false;
    po.Register("cross-validate", &cross_validate,
                "Do cross validation without update the weights");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    std::string top_ae_nnet;
    po.Register("top-ae-nnet", &top_ae_nnet, "Top level RBMs converted AE");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2);

    std::string target_model_filename;
    target_model_filename = po.GetOptArg(3);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet rbm_transf;
    if (feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    Nnet top_rbm_ae;
    if(top_ae_nnet != ""){
      top_rbm_ae.Read(top_ae_nnet);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kLinRbm);
    LinRbm &rbm = dynamic_cast<LinRbm&>(*nnet.Layer(0));

    rbm.SetLearnRate(learn_rate);
    rbm.SetMomentum(momentum);
    rbm.SetL2Penalty(l2_penalty);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Cache cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    CuRand<BaseFloat> cu_rand;
    Mse mse;

    CuMatrix<BaseFloat> feats, feats_transf, pos_vis, pos_hid, neg_vis, neg_hid, top_ae_out;
    CuMatrix<BaseFloat> dummy_mse_mat;
    std::vector<int32> dummy_cache_vec;

    Timer tim;
    double time_next = 0;
    if (!cross_validate) {
      KALDI_LOG<< "RBM TRAINING STARTED";
    } else {
      KALDI_LOG << "RBM CROSS VALIDATION STARTED";
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
        rbm_transf.Feedforward(feats, &feats_transf);
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
          << (cache.Randomized() ? "[RND]" : "[NO-RND]")
          << " segments: " << num_done
          << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&pos_vis, &dummy_cache_vec);

        // TRAIN with CD1
        // forward pass
        rbm.Propagate(pos_vis, &pos_hid);
        // possible forward through the top RBMs
        top_rbm_ae.Feedforward(pos_hid, &top_ae_out);
        // alter the hidden values, so we can generate negative example
        if (rbm.HidType() == Rbm::BERNOULLI) {
          cu_rand.BinarizeProbs(top_ae_out, &neg_hid);
        } else {
          // assume Rbm::GAUSSIAN
          neg_hid.CopyFromMat(top_ae_out);
          cu_rand.AddGaussNoise(&neg_hid);
        }
        // reconstruct pass
        rbm.Reconstruct(neg_hid, &neg_vis);
        // propagate negative examples
        rbm.Propagate(neg_vis, &neg_hid);
        if (!cross_validate) {
          // update step
          rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
        }
        // evaluate mean square error
        mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

        tot_t += pos_vis.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done())
        break;
    }

    if (!cross_validate) {
      nnet.Write(target_model_filename, binary);
    }

    std::cout << "\n" << std::flush;

    if (cross_validate) {
      KALDI_LOG<< "RBM CROSS VALIDATION ";
    } else {
      KALDI_LOG<< "RBM TRAINING FINISHED ";
    }
    KALDI_LOG<< tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files.";

    KALDI_LOG<< mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

