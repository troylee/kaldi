// nnetbin/weight-tied-ae-train.cc

//
// The AE weights are tied in a reflection way, i.e. the first layer is tied to the last layer
//

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-biasedlinearity.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
            "Usage:  weight-tied-ae-train [options] <model-in> <feature-rspecifier> <targets-rspecifier> [<model-out>]\n"
            "e.g.: \n"
            " weight-tied-ae-train nnet.init scp:train.scp ark:targets.scp nnet.iter1\n";

    ParseOptions po(usage);
    bool binary = false,
        crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string learn_factors = "";
    po.Register(
        "learn-factors",
        &learn_factors,
        "Learning factor for each updatable layer, separated by ',' and work together with learn rate");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    bool average_grad = false;
    po.Register("average-grad", &average_grad,
                "Whether to average the gradient in the bunch");

    po.Read(argc, argv);

    if (po.NumArgs() != 4 - (crossvalidate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        targets_rspecifier = po.GetArg(3);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    if(learn_factors == ""){
      nnet.SetLearnRate(learn_rate, NULL);
    }else{
      nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    }
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);
    nnet.SetAverageGrad(average_grad);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);

    CacheTgtMat cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Mse mse;

    CuMatrix<BaseFloat> feats, feats_transf, targets, nnet_in, nnet_out,
        nnet_tgt, glob_err;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache, 
      // both reader are sequential, not to be too memory hungry,
      // the scp lists must be in the same order 
      // we run the loop over targets, skipping features with no targets
      while (!cache.Full() && !feature_reader.Done() && !targets_reader.Done()) {
        // get the keys
        std::string tgt_key = targets_reader.Key();
        std::string fea_key = feature_reader.Key();
        // skip feature matrix with no targets
        while (fea_key != tgt_key) {
          KALDI_WARN<< "No targets for: " << fea_key;
          num_no_tgt_mat++;
          if (!feature_reader.Done()) {
            feature_reader.Next();
            fea_key = feature_reader.Key();
          }
        }
          // now we should have a pair
        if (fea_key == tgt_key) {
          // get feature tgt_mat pair
          const Matrix<BaseFloat> &fea_mat = feature_reader.Value();
          const Matrix<BaseFloat> &tgt_mat = targets_reader.Value();
          // chech for dimension
          if (tgt_mat.NumRows() != fea_mat.NumRows()) {
            KALDI_WARN<< "Alignment has wrong size "<< (tgt_mat.NumRows()) << " vs. "<< (fea_mat.NumRows());
            num_other_error++;
            continue;
          }
            // push features/targets to GPU
          feats.CopyFromMat(fea_mat);
          targets.CopyFromMat(tgt_mat);
          // possibly apply feature transform
          nnet_transf.Feedforward(feats, &feats_transf);
          // add to cache
          cache.AddData(feats_transf, targets);
          num_done++;
        }
        Timer t_features;
        feature_reader.Next();
        targets_reader.Next();
        time_next += t_features.Elapsed();
      }
      // randomize
      if (!crossvalidate) {
        cache.Randomize();
      }
      // report
      std::cerr << "Cache #" << ++num_cache << " "
          << (cache.Randomized() ? "[RND]" : "[NO-RND]")
          << " segments: " << num_done
          << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&nnet_in, &nnet_tgt);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);
        mse.Eval(nnet_out, nnet_tgt, &glob_err);
        if (!crossvalidate) {
          nnet.Backpropagate(glob_err, NULL);

          // Do the weight tying
          int32 tot_layers = nnet.LayerCount();
          int32 num_bls = 0;
          for (int32 i = 0; i < tot_layers; ++i) {
            if ((nnet.Layer(i))->GetType() == Component::kBiasedLinearity) {
              ++num_bls;
            }
          }
          if (num_bls % 2 != 0) {
            KALDI_WARN
                << "Odd number of <biasedlinearity> layers, weight tying is not applied!";
              } else {
                for (int32 i = 0, j = tot_layers - 1; i < j;) {
                  if (nnet.Layer(i)->GetType() != Component::kBiasedLinearity) {
                    ++i;
                  }
                  if (nnet.Layer(j)->GetType() != Component::kBiasedLinearity) {
                    --j;
                  }
                  if(nnet.Layer(i)->GetType() == Component::kBiasedLinearity && nnet.Layer(j)->GetType() == Component::kBiasedLinearity) {
                    BiasedLinearity& bl1 = dynamic_cast<BiasedLinearity&>(*nnet.Layer(i));
                    BiasedLinearity& bl2 = dynamic_cast<BiasedLinearity&>(*nnet.Layer(j));
                    bl1.SetLinearityWeight(bl2.GetLinearityWeight(), true);
                    ++i;
                    --j;
                  }
                }
              }
            }
        tot_t += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done())
        break;
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_tgt_mat
    << " with no tgt_mats, " << num_other_error
    << " with other errors.";

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
