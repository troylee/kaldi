// nnetbin/nnet2-train-two-tasks-frmshuff.cc

/*
 * Created on: Sep 20, 2013
 *     Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *     Training Nnet with two objectives.
 *     Multi-task Nnet.
 *
 *     Totally 3 nnets are involved, one for shared representation, and two for the two different
 *     tasks respectively.
 *
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-xent-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {

  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Perform iteration of two-objective Neural Network training by stochastic gradient descent.\n"
            "Usage:  nnet-train-two-tasks-frmshuff [options] <shared-nnet-in> <xent-nnet-in> "
            "<mse-nnet-in> <feature-rspecifier> <xent-align-rspecifier> <mse-tgtmat-rspecifier> "
            "[<shared-nnet-out> <xent-nnet-out> <mse-nnet-out>]\n"
            "e.g.: \n"
            " nnet-train-two-task-frmshuff shared.init xent.init mse.init scp:train.scp ark:xent.pdf "
            "ark:mse.scp shared.iter1 xent.iter1 mse.iter1\n";

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

    std::string learn_factors_shared = "", learn_factors_xent = "",
        learn_factors_mse = "";
    po.Register(
        "learn-factors-shared",
        &learn_factors_shared,
        "Learning factor for each updatable layer in the shared Nnet, separated by ',' and work together with learn rate");
    po.Register("learn-factors-xent", &learn_factors_xent,
                "Learning factors for task1");
    po.Register("learn-factors-mse", &learn_factors_mse,
                "Learning factors for task2");

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

    BaseFloat error_weight_xent = 0.5, error_weight_mse = 0.5;
    po.Register("error-weight-xent", &error_weight_xent, "Weight for Xent errors backpropagated to shared NNet");
    po.Register("error-weight-mse", &error_weight_mse, "Weight for MSE errors backpropagated to shared NNet");

    po.Read(argc, argv);

    if (po.NumArgs() != 9 - (crossvalidate ? 3 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string shared_nnet_filename = po.GetArg(1),
        xent_nnet_filename = po.GetArg(2),
        mse_nnet_filename = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        xent_align_rspecifier = po.GetArg(5),
        mse_tgtmat_rspecifier = po.GetArg(6);

    std::string shared_nnet_out, xent_nnet_out, mse_nnet_out;
    if (!crossvalidate) {
      shared_nnet_out = po.GetArg(7);
      xent_nnet_out = po.GetArg(8);
      mse_nnet_out = po.GetArg(9);
    }

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet_shared, nnet_xent, nnet_mse;
    nnet_shared.Read(shared_nnet_filename);
    nnet_xent.Read(xent_nnet_filename);
    nnet_mse.Read(mse_nnet_filename);

    if (learn_factors_shared == "") {
      nnet_shared.SetLearnRate(learn_rate, NULL);
    } else {
      nnet_shared.SetLearnRate(learn_rate, learn_factors_shared.c_str());
    }
    nnet_shared.SetMomentum(momentum);
    nnet_shared.SetL2Penalty(l2_penalty);
    nnet_shared.SetL1Penalty(l1_penalty);
    nnet_shared.SetAverageGrad(average_grad);

    if (learn_factors_xent == "") {
      nnet_xent.SetLearnRate(learn_rate, NULL);
    } else {
      nnet_xent.SetLearnRate(learn_rate, learn_factors_xent.c_str());
    }
    nnet_xent.SetMomentum(momentum);
    nnet_xent.SetL2Penalty(l2_penalty);
    nnet_xent.SetL1Penalty(l1_penalty);
    nnet_xent.SetAverageGrad(average_grad);

    if (learn_factors_mse == "") {
      nnet_mse.SetLearnRate(learn_rate, NULL);
    } else {
      nnet_mse.SetLearnRate(learn_rate, learn_factors_mse.c_str());
    }
    nnet_mse.SetMomentum(momentum);
    nnet_mse.SetL2Penalty(l2_penalty);
    nnet_mse.SetL1Penalty(l1_penalty);
    nnet_mse.SetAverageGrad(average_grad);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader mse_tgtmat_reader(mse_tgtmat_rspecifier);
    RandomAccessInt32VectorReader xent_align_reader(xent_align_rspecifier);

    CacheXentTgtMat cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;
    Mse mse;

    CuMatrix<BaseFloat> feats, feats_transf, targets, nnet_in, nnet_out,
        nnet_xent_out, nnet_mse_out, nnet_xent_out_raw, nnet_mse_out_raw,
        nnet_tgt, glob_err;
    CuMatrix<BaseFloat> nnet_out_shared, nnet_out_xent, nnet_out_mse,
        xent_err, mse_err, shared_err_from_xent, shared_err;
    std::vector<int32> nnet_labs;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_align = 0, num_no_tgt_mat = 0, num_other_error =
        0, num_cache = 0;
    while (1) {
      // fill the cache, 
      // both reader are sequential, not to be too memory hungry,
      // the scp lists must be in the same order 
      // we run the loop over targets, skipping features with no targets
      while (!cache.Full() && !feature_reader.Done()
          && !mse_tgtmat_reader.Done()) {
        // get the keys
        std::string tgt_key = mse_tgtmat_reader.Key();
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
          const Matrix<BaseFloat> &tgt_mat = mse_tgtmat_reader.Value();
          // chech for dimension
          if (tgt_mat.NumRows() != fea_mat.NumRows()) {
            KALDI_WARN<< "Target mat has wrong size "<< (tgt_mat.NumRows()) << " vs. "<< (fea_mat.NumRows());
            num_other_error++;
            continue;
          }

          if (!xent_align_reader.HasKey(fea_key)) {
            ++num_no_align;
            KALDI_WARN<< "No alignments for: " << fea_key;
            continue;
          }

          const std::vector<int32> &labs = xent_align_reader.Value(fea_key);
          if (labs.size() != fea_mat.NumRows()) {
            ++num_other_error;
            KALDI_WARN<< "Aligment has wrong size " << (labs.size()) << " vs. " << (fea_mat.NumRows());
            continue;
          }

          // push features/targets to GPU
          feats.CopyFromMat(fea_mat);
          targets.CopyFromMat(tgt_mat);
          // possibly apply feature transform
          nnet_transf.Feedforward(feats, &feats_transf);
          // add to cache
          cache.AddData(feats_transf, labs, targets);
          num_done++;
        }
        Timer t_features;
        feature_reader.Next();
        mse_tgtmat_reader.Next();
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
        cache.GetBunch(&nnet_in, &nnet_labs, &nnet_tgt);
        // train 
        nnet_shared.Propagate(nnet_in, &nnet_out_shared);
        nnet_xent.Propagate(nnet_out_shared, &nnet_out_xent);
        nnet_mse.Propagate(nnet_out_shared, &nnet_out_mse);

        xent.EvalVec(nnet_out_xent, nnet_labs, &xent_err);
        mse.Eval(nnet_out_mse, nnet_tgt, &mse_err);

        if (!crossvalidate) {
          // Backpropagate errors through task nnets
          nnet_xent.Backpropagate(xent_err, &shared_err_from_xent);
          nnet_mse.Backpropagate(mse_err, &shared_err);

          // average the two errors
          shared_err.AddMat(error_weight_xent, shared_err_from_xent, error_weight_mse);

          // Backpropagate errors through the shared nnet
          nnet_shared.Backpropagate(shared_err, NULL);
        }
        tot_t += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done())
        break;
    }

    if (!crossvalidate) {
      nnet_shared.Write(shared_nnet_out, binary);
      nnet_xent.Write(xent_nnet_out, binary);
      nnet_mse.Write(mse_nnet_out, binary);
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_align
    << " with no alignments, " << num_no_tgt_mat
    << " with no tgt_mats, " << num_other_error
    << " with other errors.";

    if(!crossvalidate) KALDI_LOG << "Save Xent Nnet to " << xent_nnet_out << ".\n";
    KALDI_LOG<< xent.Report();
    if(!crossvalidate) KALDI_LOG << "Save Mse Nnet to " << mse_nnet_out << ".\n";
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
