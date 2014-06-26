// nnetbin/nnet-hidmask-mse-hardlab-perutt.cc

/*
 * Apply mask to hidden layers, train the hidden mask estimator with
 * parallel data so no need to save out the masks.
 *
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
            "Usage:  nnet-train-mse-tgtmat-frmshuff [options] <model-in> <frontend-model-in>"
            " <noisy-feature-rspecifier> <clean-feature-rspecifier> [<model-out>]\n"
            "e.g.: \n"
            " nnet-train-mse-tgtmat-frmshuff nnet.init scp:train.scp ark:targets.scp nnet.iter1\n";

    ParseOptions po(usage);

    bool binarize_mask = false;
    po.Register("binarize-mask", &binarize_mask,
                "Binarize the hidden mask or not");

    BaseFloat binarize_threshold = 0.5;
    po.Register("binarize-threshold", &binarize_threshold,
                "Threshold value to binarize mask");

    BaseFloat alpha = 3.0;
    po.Register("alpha", &alpha, "Alpha value for hidden masking");

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

    if (po.NumArgs() != 5 - (crossvalidate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        frontend_model_filename = po.GetArg(2),
        noisyfeats_rspecifier = po.GetArg(3),
        cleanfeats_rspecifier = po.GetArg(4);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(5);
    }

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet_frontend;
    nnet_frontend.Read(frontend_model_filename);

    Nnet nnet;
    nnet.Read(model_filename);

    if (learn_factors == "") {
      nnet.SetLearnRate(learn_rate, NULL);
    } else {
      nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    }
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);
    nnet.SetAverageGrad(average_grad);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader noisyfeats_reader(noisyfeats_rspecifier);
    SequentialBaseFloatMatrixReader cleanfeats_reader(cleanfeats_rspecifier);

    CacheTgtMat cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Mse mse;

    CuMatrix<BaseFloat> noisy_feats, noisy_transf, noisy_hids,
        clean_feats, clean_transf, clean_hids, hid_masks;
    CuMatrix<BaseFloat> nnet_in, nnet_out, nnet_tgt, glob_err;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache, 
      // both reader are sequential, not to be too memory hungry,
      // the scp lists must be in the same order 
      // we run the loop over targets, skipping features with no targets
      while (!cache.Full() && !noisyfeats_reader.Done()
          && !cleanfeats_reader.Done()) {
        // get the keys
        std::string noisy_key = noisyfeats_reader.Key();
        std::string clean_key = cleanfeats_reader.Key();
        // stopping if the keys do not match
        if (noisy_key != clean_key) {
          KALDI_ERR<< "Key mismatches for parallel data: " << noisy_key << " vs. " << clean_key;
        }
          // now we should have a pair
        if (noisy_key == clean_key) {
          // get feature pari
          const Matrix<BaseFloat> &noisy_mats = noisyfeats_reader.Value();
          const Matrix<BaseFloat> &clean_mats = cleanfeats_reader.Value();
          if (noisy_mats.NumRows() != clean_mats.NumRows()
              || noisy_mats.NumCols() != clean_mats.NumCols()) {
            KALDI_WARN<< "Feature mismatch: " << noisy_key;
            num_other_error++;
            continue;
          }

            // push features/targets to GPU
          noisy_feats.CopyFromMat(noisy_mats);
          clean_feats.CopyFromMat(clean_mats);
          // possibly apply feature transform
          nnet_transf.Feedforward(noisy_feats, &noisy_transf);
          nnet_transf.Feedforward(clean_feats, &clean_transf);
          // compute mask
          nnet_frontend.Feedforward(noisy_transf, &noisy_hids);
          nnet_frontend.Feedforward(clean_transf, &clean_hids);
          hid_masks.CopyFromMat(noisy_hids);
          hid_masks.AddMat(-1.0, clean_hids, 1.0);
          hid_masks.Power(2.0);
          hid_masks.Scale(-1.0 * alpha);
          hid_masks.ApplyExp();
          if (binarize_mask)
            hid_masks.Binarize(binarize_threshold);
          // add to cache
          cache.AddData(noisy_transf, hid_masks);
          num_done++;
        }
        Timer t_features;
        noisyfeats_reader.Next();
        cleanfeats_reader.Next();
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
        }
        tot_t += nnet_in.NumRows();
      }

      // stop training when no more data
      if (noisyfeats_reader.Done() || cleanfeats_reader.Done())
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
