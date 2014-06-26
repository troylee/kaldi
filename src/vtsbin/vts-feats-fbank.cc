/*
 * vtsbin/vts-feats.cc
 *
 *  Created on: Nov 30, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Feature based VTS, with given noise estimation.
 *
 * [Thesis] Pedro J. Moreno, Speech Recognition in Noisy Environments, 1996, p 91
 *
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/diag-gmm-normal.h"
#include "vts/vts-first-order.h"
#include "feat/feature-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "VTS feature compensation using Diagonal GMM-based model with given noise estimation.\n"
        "Usage:  vts-feats-fbank [options] gmm-model-in noisy-mfcc-rspecifier noisy-fbank-rspecifier noiseparams-rspecifier"
        " clean-features-wspecifier\n"
        "Note: Features are FBank_E_D_A.\n";
    DeltaFeaturesOptions opts;
    ParseOptions po(usage);

    bool have_energy = true;
    po.Register("have-energy", &have_energy,
        "Whether the feature has energy term, it will not be compensated");

    bool update_dynamic = true;
    po.Register("update-dynamic", &update_dynamic,
        "Whether to update the dynamic parameters. If not, the noisy version will be used");

    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;

    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string gmm_model_rxfilename = po.GetArg(1),
        noisy_mfcc_rspecifier = po.GetArg(2),
        noisy_fbank_rspecifier = po.GetArg(3),
        noiseparams_rspecifier = po.GetArg(4),
        clean_feature_wspecifier = po.GetArg(5);

    // the clean speech GMM model
    DiagGmm clean_gmm;
    {
      bool binary;
      Input ki(gmm_model_rxfilename, &binary);
      clean_gmm.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader noisy_mfcc_reader(noisy_mfcc_rspecifier);
    RandomAccessBaseFloatMatrixReader noisy_fbank_reader(noisy_fbank_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noiseparams_rspecifier);
    BaseFloatMatrixWriter clean_feature_writer(clean_feature_wspecifier);

    int num_success = 0, num_fail = 0;

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    Timer timer;

    for (; !noisy_mfcc_reader.Done(); noisy_mfcc_reader.Next()) {
      std::string key = noisy_mfcc_reader.Key();
      Matrix<BaseFloat> noisy_mfcc(noisy_mfcc_reader.Value());
      noisy_mfcc_reader.FreeCurrent();

      KALDI_VLOG(1)<< "Current utterance: " << key;

      if (noisy_mfcc.NumRows() == 0) {
        KALDI_WARN<< "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
      }

      int32 feat_dim = noisy_mfcc.NumCols();
      if (feat_dim != 39) {
        KALDI_ERR<< "Current only support FBANK_D_A or FBANK_E_D_A!";
      }

      /************************************************
       Extract the noise parameters
       *************************************************/

      Vector<double> mfcc_mu_h(noiseparams_reader.Value(key + "_mu_h"));
      Vector<double> mfcc_mu_z(noiseparams_reader.Value(key + "_mu_z"));
      Vector<double> mfcc_var_z(noiseparams_reader.Value(key + "_var_z"));

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG<< "Additive Noise Mean: " << mfcc_mu_z;
        KALDI_LOG << "Additive Noise Covariance: " << mfcc_var_z;
        KALDI_LOG << "Convoluational Noise Mean: " << mfcc_mu_h;
      }

        // only static part
      Vector<double> fbank_mu_h(num_fbank), fbank_mu_z(num_fbank);
      fbank_mu_h.AddMatVec(1.0, inv_dct_mat, kNoTrans,
                           SubVector<double>(mfcc_mu_h, 0, num_cepstral), 0.0);
      fbank_mu_z.AddMatVec(1.0, inv_dct_mat, kNoTrans,
                           SubVector<double>(mfcc_mu_z, 0, num_cepstral), 0.0);

      /************************************************
       Extract the noise parameters
       *************************************************/
      if (!noisy_fbank_reader.HasKey(key)) {
        KALDI_ERR<< "Cannot find corresponding FBank features!";
      }

      Matrix<BaseFloat> noisy_fbank(noisy_fbank_reader.Value(key));

      if (noisy_fbank.NumCols()
          != (have_energy ? num_fbank + 1 : num_fbank) * 3) {
        KALDI_ERR<< "Current only support FBANK_D_A or FBANK_E_D_A!";
      }

        /************************************************
         Compensate the model
         *************************************************/

      DiagGmm noise_gmm;
      // Initialize with the clean speech model
      noise_gmm.CopyFromDiagGmm(clean_gmm);

      std::vector<Matrix<double> > Jx(clean_gmm.NumGauss()), Jz(
          clean_gmm.NumGauss());  // not necessary for compensation only
      CompensateDiagGmm(mfcc_mu_h, mfcc_mu_z, mfcc_var_z, num_cepstral,
                        num_fbank, dct_mat, inv_dct_mat, noise_gmm, Jx, Jz);

      /************************************************
       Reconstruct the clean features
       *************************************************/

      DiagGmmNormal clean_ngmm(clean_gmm);
      // Gaussian component based residual y = x + com_g
      Matrix<double> com_g(clean_gmm.NumGauss(), num_fbank);
      Vector<double> tmp_mfcc(num_cepstral), tmp_fbank(num_fbank);
      for (int32 gid = 0; gid < clean_gmm.NumGauss(); ++gid) {

        tmp_fbank.CopyFromVec(fbank_mu_z);  // n
        tmp_fbank.AddMatVec(-1.0, inv_dct_mat, kNoTrans,
            SubVector<double>(clean_ngmm.means_.Row(gid), 0, num_cepstral), 1.0);  // n-x
        tmp_fbank.AddVec(-1.0, fbank_mu_h);  // n-x-h

        tmp_fbank.ApplyExp();
        tmp_fbank.Add(1.0);
        tmp_fbank.ApplyLog();

        tmp_fbank.AddVec(1.0, fbank_mu_h);

        com_g.CopyRowFromVec(tmp_fbank, gid);
      }

      // compute clean features
      Vector<BaseFloat> post(clean_gmm.NumGauss());
      Matrix<double> tmp_mat(clean_gmm.NumGauss(), num_fbank);
      Matrix<BaseFloat> delta_static_features(noisy_fbank.NumRows(),
                                              num_fbank, kSetZero);
      for (int32 fid = 0; fid < noisy_fbank.NumRows(); ++fid) {
        noise_gmm.ComponentPosteriors(noisy_mfcc.Row(fid), &post);
        tmp_mat.CopyFromMat(com_g);
        tmp_mat.MulRowsVec(Vector<double>(post));

        tmp_fbank.AddRowSumMat(-1.0, tmp_mat, 0.0);

        delta_static_features.CopyRowFromVec(Vector<BaseFloat>(tmp_fbank), fid);
      }

      if (update_dynamic) {
        // recompute the dynamic parameters
        KALDI_ERR<< "Not implemented yet!";
      } else {
        SubMatrix<BaseFloat> noisy_static(noisy_fbank, 0,
            noisy_fbank.NumRows(), 0,
            num_fbank);
        noisy_static.AddMat(1.0, delta_static_features, kNoTrans);

        clean_feature_writer.Write(key, noisy_fbank);
        ++num_success;
      }

      if (num_success % 100 == 0) {
        KALDI_LOG<< "Done " << num_success << " files.";
      }

    }

    KALDI_LOG<< "Done " << num_success << " utterances, failed for "
    << num_fail;

    return (num_success != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

