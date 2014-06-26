/*
 * nnet/nnet-hidmask-train-frmshuff.cc
 *
 *  Created on: Oct 1, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-xent-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
            "Usage: nnet-hidmask-train-frmshuff [options] <frontend-model-in> <backend-model-in>"
            " <noisy-feature-rspecifier> <clean-feature-rspecifier> <alignments-rspecifier> "
            "[<frontend-model-in> <backend-model-out>]\n"
            "e.g.: \n "
            " nnet-hidmask-train-frmshuff front.nnet back.nnet scp:train_noisy.scp scp:train_clean.scp "
            "ark:train.ali front.iter1 back.iter1\n";

    ParseOptions po(usage);
    bool binary = false,
        cross_validate = false,
        randomize = true,
        binarize_mask = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &cross_validate,
                "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");
    po.Register("binarize-mask", &binarize_mask,
                "Binarize the hidden mask or not");

    BaseFloat binarize_threshold = 0.5;
    po.Register("binarize-threshold", &binarize_threshold,
                "Threshold value to binarize mask");

    BaseFloat alpha = 3.0;
    po.Register("alpha", &alpha, "Alpha value for hidden masking");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;
    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad,
                "Whether to average the gradient in the bunch");

    std::string learn_factors_front = "", learn_factors_back = "";
    po.Register("learn-factors-front", &learn_factors_front,
                "Learning factors for front-end Nnet");
    po.Register("learn-factors-back", &learn_factors_back,
                "Learning factors for back-end Nnet");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    po.Read(argc, argv);

    if (po.NumArgs() != 7 - (cross_validate ? 2 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_frontend_filename = po.GetArg(1),
        model_backend_filename = po.GetArg(2),
        noisyfeats_rspecifier = po.GetArg(3),
        cleanfeats_rspecifier = po.GetArg(4),
        alignments_rspecifier = po.GetArg(5);

    std::string target_frontend_filename, target_backend_filename;
    if (!cross_validate) {
      target_frontend_filename = po.GetArg(6);
      target_backend_filename = po.GetArg(7);
    }

    /*
     * Load Nnets
     */
    // feature transformation nnet
    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    // The hidden masking happends between the front-end and back-end DNN
    Nnet nnet_frontend;
    nnet_frontend.Read(model_frontend_filename);
    if (learn_factors_front == "") {
      nnet_frontend.SetLearnRate(learn_rate, NULL);
    } else {
      nnet_frontend.SetLearnRate(learn_rate, learn_factors_front.c_str());
    }
    nnet_frontend.SetMomentum(momentum);
    nnet_frontend.SetL2Penalty(l2_penalty);
    nnet_frontend.SetL1Penalty(l1_penalty);
    nnet_frontend.SetAverageGrad(average_grad);

    Nnet nnet_backend;
    nnet_backend.Read(model_backend_filename);
    if (learn_factors_back == "") {
      nnet_backend.SetLearnRate(learn_rate, NULL);
    } else {
      nnet_backend.SetLearnRate(learn_rate, learn_factors_back.c_str());
    }
    nnet_backend.SetMomentum(momentum);
    nnet_backend.SetL2Penalty(l2_penalty);
    nnet_backend.SetL1Penalty(l1_penalty);
    nnet_backend.SetAverageGrad(average_grad);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader noisyfeats_reader(noisyfeats_rspecifier);
    SequentialBaseFloatMatrixReader cleanfeats_reader(cleanfeats_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    CacheXentTgtMat cache;  // using the tgtMat to save the clean feats
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    CuMatrix<BaseFloat> noisy_feats, clean_feats, noisy_feats_transf,
        clean_feats_transf;

    CuMatrix<BaseFloat> front_noisy_in, front_noisy_out,
        front_clean_in, front_clean_out, nnet_out, hid_masks,
        glob_err, front_err;
    std::vector<int32> nnet_labs;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (cross_validate? "CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_align = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !noisyfeats_reader.Done()
          && !cleanfeats_reader.Done()) {
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

          if (!alignments_reader.HasKey(noisy_key)) {
            ++num_no_align;
            KALDI_WARN<< "No alignments for: " << noisy_key;
            continue;
          }

          const std::vector<int32> &labs = alignments_reader.Value(noisy_key);
          if (labs.size() != noisy_mats.NumRows()) {
            ++num_other_error;
            KALDI_WARN<< "Alignment has wrong size " << (labs.size()) << " vs. " << (noisy_mats.NumRows());
            continue;
          }

          // push features to GPU
          noisy_feats.CopyFromMat(noisy_mats);
          clean_feats.CopyFromMat(clean_mats);
          // possibly apply feature transform
          nnet_transf.Feedforward(noisy_feats, &noisy_feats_transf);
          nnet_transf.Feedforward(clean_feats, &clean_feats_transf);
          // add to cache
          cache.AddData(noisy_feats_transf, labs, clean_feats_transf);
          num_done++;
        }
        Timer t_features;
        noisyfeats_reader.Next();
        cleanfeats_reader.Next();
        time_next += t_features.Elapsed();
      }
      // randomize
      if (!cross_validate) {
        cache.Randomize();
      }
      // report
      std::cerr << "Cache #" << ++num_cache << " "
          << (cache.Randomized() ? "[RND]" : "[NO-RND]")
          << " segments: " << num_done
          << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of features pairs
        cache.GetBunch(&front_noisy_in, &nnet_labs, &front_clean_in);

        if (!cross_validate) {  // do clean first so that the buffers are overwrittend by noisy features
          nnet_frontend.Propagate(front_clean_in, &front_clean_out);
        }

        // forward through nnet_hidmask
        nnet_frontend.Propagate(front_noisy_in, &front_noisy_out);

        if (!cross_validate) {
          // compute hid masks
          hid_masks.CopyFromMat(front_noisy_out);
          hid_masks.AddMat(-1.0, front_clean_out, 1.0);
          hid_masks.Power(2.0);
          hid_masks.Scale(-1.0 * alpha);
          hid_masks.ApplyExp();

          if (binarize_mask)
            hid_masks.Binarize(binarize_threshold);

          // apply masks to hidden acts
          front_noisy_out.MulElements(hid_masks);
        }

        // forward through backend nnet
        nnet_backend.Propagate(front_noisy_out, &nnet_out);

        // evaluate
        xent.EvalVec(nnet_out, nnet_labs, &glob_err);

        if (!cross_validate) {
          nnet_backend.Backpropagate(glob_err, &front_err);

          front_err.MulElements(hid_masks);
          nnet_frontend.Backpropagate(front_err, NULL);

        }
        tot_t += nnet_labs.size();

      }

      // stop training when no more data
      if (noisyfeats_reader.Done() || cleanfeats_reader.Done())
        break;
    }

    if (!cross_validate) {
      nnet_frontend.Write(target_frontend_filename, binary);
      nnet_backend.Write(target_backend_filename, binary);
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (cross_validate? "CROSSVALIDATE":"TRAINING") << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_align
    << " with no alignments, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< xent.Report();

#if HAVE_CUDA == 1
        CuDevice::Instantiate().PrintProfile();
#endif

      } catch (const std::exception &e) {
        std::cerr << e.what();
        return -1;
      }

    }

