/*
 * vts-compute-mono-post.cc
 *
 *  Created on: Oct 22, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "util/timer.h"
#include "gmm/diag-gmm-normal.h"
#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Compute monophone state log-posterior from triphone GMM-based model with 1st order VTS model compensation.\n"
        "(outputs matrices of log-posteriors indexed by (frame, monophone-state)\n"
        "Usage: vts-compute-mono-post [options] model-in mono2tri-matrix features-rspecifier noiseparams-rspecifier post-wspecifier\n";

    ParseOptions po(usage);
    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;
    bool take_log = true;

    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");
    po.Register("take-log", &take_log, "Output log posterior probabilities or not");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        mono2tri_matrix_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        noiseparams_rspecifier = po.GetArg(4),
        loglikes_wspecifier = po.GetArg(5);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;  // not needed for computation, just in case to save the noisy model
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Matrix<BaseFloat> mono2tri;
    {
      bool mat_binary;
      Input matrix_input(mono2tri_matrix_filename, &mat_binary);
      mono2tri.Read(matrix_input.Stream(), mat_binary);  // row: mono-state, col: pdf
    }
    KALDI_LOG << "mono2tri matrix size: [" << mono2tri.NumRows() << ", "
        << mono2tri.NumCols() << "].";

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    if (g_kaldi_verbose_level >= 2) {
      KALDI_LOG << "DCT Transform: " << dct_mat;
      KALDI_LOG << "Inverse DCT: " << inv_dct_mat;
    }

    BaseFloatMatrixWriter loglikes_writer(loglikes_wspecifier);
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noiseparams_rspecifier);

    int32 num_done = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &features(feature_reader.Value());
      Vector<BaseFloat> loglikes(am_gmm.NumPdfs());
      Matrix<BaseFloat> logposts(features.NumRows(), mono2tri.NumRows());

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Current utterance: " << key;
      }

      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        continue;
      }

      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR
            << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
      }

      int feat_dim = features.NumCols();
      if (feat_dim != 39) {
        KALDI_ERR
            << "Could not decode the features, only 39D MFCC_0_D_A is supported!";
      }

      /************************************************
       Extract the noise parameters
       *************************************************/

      Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
      Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
      Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
      }
      /************************************************
       Compensate the model
       *************************************************/

      AmDiagGmm noise_am_gmm;
      // Initialize with the clean speech model
      noise_am_gmm.CopyFromAmDiagGmm(am_gmm);

      std::vector<Matrix<double> > Jx(am_gmm.NumGauss()), Jz(am_gmm.NumGauss());  // not necessary for compensation only
      CompensateModel(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat,
                      inv_dct_mat, noise_am_gmm, Jx, Jz);

      if (g_kaldi_verbose_level >= 20) {
        // verify the model is indeed changed
        {
          Output ko("noise_model.txt", false);
          trans_model.Write(ko.Stream(), false);
          noise_am_gmm.Write(ko.Stream(), false);
        }
      }

      for (int32 i = 0; i < features.NumRows(); i++) {
        std::vector<bool> flag(mono2tri.NumRows(), true);

        for (int32 j = 0; j < noise_am_gmm.NumPdfs(); j++) {
          SubVector<BaseFloat> feat_row(features, i);
          loglikes(j) = noise_am_gmm.LogLikelihood(j, feat_row);

          for (int32 k = 0; k < mono2tri.NumRows(); ++k) {
            if (mono2tri(k, j) > 0.0) {
              if (flag[k]) {
                logposts(i, k) = loglikes(j);
                flag[k] = false;
              } else {
                logposts(i, k) = LogAdd(logposts(i, k), loglikes(j));
              }
            }
          }
        }

        SubVector<BaseFloat> lp(logposts, i);
        BaseFloat offset = -lp.LogSumExp();
        lp.Add(offset);
        if(!take_log) lp.ApplyExp();
        logposts.CopyRowFromVec(lp, i);
      }
      loglikes_writer.Write(key, logposts);
      num_done++;
    }

    KALDI_LOG << "vts-compute-mono-post: computed log posteriors for "
        << num_done << " utterances.";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

