// nnetbin/nnet-train-xent-mse-frmshuff.cc

/*
 * Created on: Sep 14, 2013
 *     Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *     Training Nnet with both xent and mse output objectives.
 *     Multi-task Nnet.
 *
 *     The Nnet doesn't have <softmax> layer in the model file, the final two-objective output is handled
 *     in this program specifically.
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
            "Usage:  nnet-train-xent-mse-tgtmat-frmshuff [options] <model-in> <feature-rspecifier> "
            "<xent-alignment-rspecifier> <mse-targets-rspecifier> [<model-out>]\n"
            "e.g.: \n"
            " nnet-train-xent-mse-frmshuff nnet.init scp:train.scp ark:train.pdf ark:targets.scp nnet.iter1\n";

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

    int32 xent_dim = -1;
    po.Register(
        "xent-dim", &xent_dim,
        "The dimnsion of the Xent outupts, smaller than Nnet output dim.");

    po.Read(argc, argv);

    if (po.NumArgs() != 5 - (crossvalidate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_rspecifier = po.GetArg(3),
        targets_rspecifier = po.GetArg(4);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(5);
    }

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

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

    if (xent_dim < 0 || xent_dim > nnet.OutputDim()) {
      KALDI_ERR<< "Invalide Xent dimension: " << xent_dim << ", should be in range [0, " << nnet.OutputDim() <<"].\n";
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignment_rspecifier);

    CacheXentTgtMat cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;
    Mse mse;

    CuMatrix<BaseFloat> feats, feats_transf, targets, nnet_in, nnet_out,
        nnet_xent_out, nnet_mse_out, nnet_xent_out_raw, nnet_mse_out_raw,
        nnet_tgt, glob_err, xent_err, mse_err;
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
            KALDI_WARN<< "Target mat has wrong size "<< (tgt_mat.NumRows()) << " vs. "<< (fea_mat.NumRows());
            num_other_error++;
            continue;
          }

          if (!alignments_reader.HasKey(fea_key)) {
            ++num_no_align;
            KALDI_WARN<< "No alignments for: " << fea_key;
            continue;
          }

          const std::vector<int32> &labs = alignments_reader.Value(fea_key);
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
        cache.GetBunch(&nnet_in, &nnet_labs, &nnet_tgt);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);

        // split the output
        nnet_xent_out_raw.CopyFromMat(nnet_out, 0, nnet_out.NumRows(), 0, xent_dim);
        nnet_mse_out_raw.CopyFromMat(nnet_out, 0, nnet_out.NumRows(), xent_dim,
                                 nnet_out.NumCols() - xent_dim);

        // applying non-liearity
        nnet_xent_out.Resize(nnet_xent_out_raw.NumRows(), nnet_xent_out_raw.NumCols());
        cu::Softmax(nnet_xent_out_raw, &nnet_xent_out);
        nnet_mse_out.Resize(nnet_mse_out_raw.NumRows(), nnet_mse_out_raw.NumCols());
        cu::Sigmoid(nnet_mse_out_raw, &nnet_mse_out);

        xent.EvalVec(nnet_xent_out, nnet_labs, &xent_err);
        mse.Eval(nnet_mse_out, nnet_tgt, &mse_err);

        if (!crossvalidate) {
          // merge the error
          glob_err.Resize(nnet_out.NumRows(), nnet_out.NumCols());
          xent_err.CopyToMat(&glob_err, 0, 0);
          mse_err.CopyToMat(&glob_err, 0, xent_dim);

          nnet.Backpropagate(glob_err, NULL);
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

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_align
    << " with no alignments, " << num_no_tgt_mat
    << " with no tgt_mats, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< xent.Report();
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
