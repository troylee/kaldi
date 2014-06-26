/*
 * vts-compute-likes.cc
 *
 *  Created on: Oct 30, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "util/timer.h"

#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Compute log-likelihoods from VTS compensated GMM-based model\n"
        "(outputs matrices of log-likelihoods indexed by (frame, pdf)\n"
        "Usage: gmm-compute-likes [options] model-in features-rspecifier noiseparams-rspecifier likes-wspecifier\n";
    ParseOptions po(usage);

    bool apply_log = true;
    po.Register("apply-log", &apply_log, "Output log-likelihoods");

    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;

    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        noiseparams_rspecifier = po.GetArg(3),
        loglikes_wspecifier = po.GetArg(4);

    AmDiagGmm am_gmm;
    {
      bool binary;
      TransitionModel trans_model;  // not needed.
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    BaseFloatMatrixWriter loglikes_writer(loglikes_wspecifier);
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noiseparams_rspecifier);

    int32 num_done = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &features(feature_reader.Value());

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

      Matrix<BaseFloat> loglikes(features.NumRows(), am_gmm.NumPdfs());
      for (int32 i = 0; i < features.NumRows(); i++) {
        for (int32 j = 0; j < am_gmm.NumPdfs(); j++) {
          SubVector<BaseFloat> feat_row(features, i);
          loglikes(i, j) = noise_am_gmm.LogLikelihood(j, feat_row);
        }
      }
      if (!apply_log) {
        loglikes.ApplyExp();
      }
      loglikes_writer.Write(key, loglikes);
      num_done++;

      if(num_done % 100 == 0){
        KALDI_LOG << num_done << ",";
      }
    }

    KALDI_LOG << "vts-compute-likes: computed" << (apply_log ? " log" : "")
        << " likelihoods for " << num_done << " utterances.";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

