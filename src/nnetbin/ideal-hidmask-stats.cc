/*
 * ideal-hidmask-stats.cc
 *
 *  Created on: Oct 26, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      This program is just for concept verification. We use parallel data to compute the ideal mask
 *      and then compute the ratio the activations are masked away.
 */

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network with ideal hidden masking.\n"
            "Usage:  ideal-hidmask-forward [options] <l1-model-in> "
            "<feature-rspecifier> <ref-feat-rspecifier>\n"
            "e.g.: \n"
            " ideal-hidmask-stats --backend-nnet=backend.nnet l1.nnet ark:features.ark "
            "ark:ref_feats.ark\n";

    ParseOptions po(usage);

    BaseFloat active_threshold = 1e-6;
    po.Register("active-threshold", &active_threshold,
                "Activations above this threshold is deemed as active");

    bool binarize_mask = false;
    po.Register("binarize-mask", &binarize_mask, "Binarize the hidden mask");

    BaseFloat binarize_threshold = 0.5;
    po.Register("binarize-threshold", &binarize_threshold,
                "Threshold to binarize the hidden mask");

    BaseFloat alpha = 1.0;
    po.Register("alpha", &alpha, "Alpha value for the hidden mask compuation");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string l1_model_filename = po.GetArg(1), feature_rspecifier =
        po.GetArg(2), ref_feats_rspecifier = po.GetArg(3);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(l1_model_filename);

    kaldi::int64 tot_t = 0, tot_act = 0, tot_act_discarded = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader ref_feats_reader(ref_feats_rspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, l1_out, nnet_out, hidmask;
    CuMatrix<BaseFloat> ref_feats, ref_feats_transf, ref_l1_out;
    Matrix<BaseFloat> l1_out_host, hidmask_host;

    Timer tim;
    KALDI_LOG<< "MLP FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done() && !ref_feats_reader.Done();
        feature_reader.Next(), ref_feats_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      std::string ref_key = ref_feats_reader.Key();
      if (key != ref_key) {
        KALDI_ERR<< "Mismatched keys: " << key << " vs. " << ref_key;
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const Matrix<BaseFloat> &ref_mat = ref_feats_reader.Value();
      if (mat.NumRows() != ref_mat.NumRows()
          || mat.NumCols() != ref_mat.NumCols()) {
        KALDI_ERR<< "Feature dimension mismatch for " << key;
      }
        //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          BaseFloat ref_val = ref_mat(r, c);
          if (val != val || ref_val != ref_val)
            KALDI_ERR<< "NaN in features of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity()
              || ref_val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in features of : " << key;
          }
        }
            // push it to gpu
      feats.CopyFromMat(mat);
      ref_feats.CopyFromMat(ref_mat);
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet_transf.Feedforward(ref_feats, &ref_feats_transf);

      nnet.Feedforward(feats_transf, &l1_out);
      nnet.Feedforward(ref_feats_transf, &ref_l1_out);

      /*
       * Do masking
       *
       */
      if (hidmask.NumRows() != l1_out.NumRows()
          || hidmask.NumCols() != l1_out.NumCols()) {
        hidmask.Resize(l1_out.NumRows(), l1_out.NumCols());
      }
      hidmask.CopyFromMat(l1_out);
      hidmask.AddMat(-1.0, ref_l1_out, 1.0);
      hidmask.Power(2.0);
      hidmask.Scale(-1.0 * alpha);
      hidmask.ApplyExp();
      if (binarize_mask)
        hidmask.Binarize(binarize_threshold);

      //download from GPU
      l1_out.CopyToMat(&l1_out_host);
      hidmask.CopyToMat(&hidmask_host);

      //accumulate statistics
      for (int32 r = 0; r < l1_out_host.NumRows(); r++) {
        for (int32 c = 0; c < l1_out_host.NumCols(); c++) {
          if (l1_out_host(r, c) > active_threshold) {
            ++tot_act;
            if (hidmask_host(r, c) <= binarize_threshold) {
              ++tot_act_discarded;
            }
          }
        }
      }

      // progress log
      if (num_done % 1000 == 0) {
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message

    KALDI_LOG<< "ACCUMULATE STATISTICS FINISHED "<< tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed();
    KALDI_LOG<< "Done " << num_done << " files";
    KALDI_LOG<< tot_act_discarded*100.0/tot_act << "% [ " << tot_act_discarded << " / " << tot_act << " ] are discarded.";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
