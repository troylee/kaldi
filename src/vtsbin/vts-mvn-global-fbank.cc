/*
 * vtsbin/vts-mvn-global-fbank.cc
 *
 *  Created on: Nov 1, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Apply global MVN normalization to the feature. The normalization
 *  statistics are got from clean features, we first compensate the
 *  normalization statistics using the noise parameters per utterance,
 *  then use these compensated, utterance specific statistics to normalize
 *  the features.
 *
 *  Only applicable to global normalization! Thus the speaker to utterance
 *  mapping is a dummy mapping from all speakers to "global".
 *
 *  Can use both the MFCC and FBank based noise estimation to compensate the FBank features.
 *  The reason to support MFCC based noise is that for GMM-HMM system MFCC based model is more
 *  robust than FBank based model. It is thus better to use MFCC based GMM for noise estimation.
 *
 *  While for other models, like DNN, we can directly use FBank.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"

#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Apply per-utterance noise compensated cepstral mean and (optionally) variance normalization\n"
        "Only global normalization is supported\n"
        "Usage: vts-mvn-global-fbank [options] cmvn-stats-rspecifier feats-rspecifier noise-rspecifier feats-wspecifier\n";

    ParseOptions po(usage);
    bool have_energy = false;
    po.Register("have-energy", &have_energy,
        "Whether the feature has energy term, it will not be compensated");

    bool norm_vars = true;
    po.Register("norm-vars", &norm_vars, "If true, normalize variances");

    std::string noise_paramkind = "mfcc";
    po.Register("noise-paramkind", &noise_paramkind,
                "Parameter kind of the noise [mfcc|fbank]");

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

    if (noise_paramkind != "mfcc" && noise_paramkind != "fbank") {
      KALDI_ERR<< "Incorrect noise parameter kind, select from 'mfcc' and 'fbank'.";
    }

    std::string cmvn_rspecifier_or_rxfilename = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string noise_rspecifier = po.GetArg(3);
    std::string feat_wspecifier = po.GetArg(4);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    Matrix<double> cmvn_stats;
    if (ClassifyRspecifier(cmvn_rspecifier_or_rxfilename, NULL, NULL)
        != kNoRspecifier) {  // reading from a Table: per-speaker or per-utt CMN/CVN.
      std::string cmvn_rspecifier = cmvn_rspecifier_or_rxfilename;

      RandomAccessDoubleMatrixReader cmvn_reader(cmvn_rspecifier);

      if (!cmvn_reader.HasKey("global")) {
        KALDI_ERR<< "No normalization statistics available for key "
        << "'global', producing no output for this utterance";
      }
      // read in the statistics
      cmvn_stats = cmvn_reader.Value("global");
    } else { // read in the statistics in normal file format
      std::string cmvn_rxfilename = cmvn_rspecifier_or_rxfilename;
      bool binary;
      Input ki(cmvn_rxfilename, &binary);
      cmvn_stats.Read(ki.Stream(), binary);
    }

    // generate the mean and covariance from the statistics
    int32 feat_dim = cmvn_stats.NumCols() - 1;
    double counts = cmvn_stats(0, feat_dim);
    Vector<double> mean(feat_dim, kSetZero), var(feat_dim, kSetZero);
    for (int32 i = 0; i < feat_dim; ++i) {
      mean(i) = cmvn_stats(0, i) / counts;
      var(i) = (cmvn_stats(1, i) / counts) - mean(i) * mean(i);
    }

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      Matrix<BaseFloat> feat(feat_reader.Value());

      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR
            << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
          }

      int feat_dim = feat.NumCols();
      if (feat_dim != (have_energy ? num_fbank + 1 : num_fbank) * 3) {
        KALDI_ERR<< "Current only support FBANK_D_A or FBANK_E_D_A!";
      }

        // compute the FBank noise estimate
      Vector<double> mu_h(num_fbank * 3, kSetZero);
      Vector<double> mu_z(num_fbank * 3, kSetZero);
      Vector<double> var_z(num_fbank * 3, kSetZero);

      if (noise_paramkind == "mfcc") {
        // MFCC noise estimate
        Vector<double> mfcc_mu_h(noiseparams_reader.Value(key + "_mu_h"));
        Vector<double> mfcc_mu_z(noiseparams_reader.Value(key + "_mu_z"));
        Vector<double> mfcc_var_z(noiseparams_reader.Value(key + "_var_z"));

        SubVector<double> static_mu_h(mu_h, 0, num_fbank);
        static_mu_h.AddMatVec(1.0, inv_dct_mat, kNoTrans,
                              SubVector<double>(mfcc_mu_h, 0, num_cepstral),
                              0.0);
        SubVector<double> static_mu_z(mu_z, 0, num_fbank);
        static_mu_z.AddMatVec(1.0, inv_dct_mat, kNoTrans,
                              SubVector<double>(mfcc_mu_z, 0, num_cepstral),
                              0.0);
        Matrix<double> tmp_mfcc(num_cepstral, num_cepstral), tmp_fbank(
            num_fbank, num_fbank);
        SubVector<double> static_var_z(var_z, 0, num_fbank);
        tmp_mfcc.SetZero();
        tmp_mfcc.CopyDiagFromVec(
            SubVector<double>(mfcc_var_z, 0, num_cepstral));
        tmp_fbank.AddMatMatMat(1.0, inv_dct_mat, kNoTrans, tmp_mfcc, kNoTrans,
                               inv_dct_mat,
                               kTrans, 0.0);
        static_var_z.CopyDiagFromMat(tmp_fbank);
        SubVector<double> delta_var_z(var_z, num_fbank, num_fbank);
        tmp_mfcc.SetZero();
        tmp_mfcc.CopyDiagFromVec(
            SubVector<double>(mfcc_var_z, num_cepstral, num_cepstral));
        tmp_fbank.AddMatMatMat(1.0, inv_dct_mat, kNoTrans, tmp_mfcc, kNoTrans,
                               inv_dct_mat,
                               kTrans, 0.0);
        delta_var_z.CopyDiagFromMat(tmp_fbank);
        SubVector<double> acc_var_z(var_z, num_fbank << 1, num_fbank);
        tmp_mfcc.SetZero();
        tmp_mfcc.CopyDiagFromVec(
            SubVector<double>(mfcc_var_z, num_cepstral << 1, num_cepstral));
        tmp_fbank.AddMatMatMat(1.0, inv_dct_mat, kNoTrans, tmp_mfcc, kNoTrans,
                               inv_dct_mat,
                               kTrans, 0.0);
        acc_var_z.CopyDiagFromMat(tmp_fbank);

      } else {  // direct FBank noise
        mu_h.CopyFromVec(noiseparams_reader.Value(key + "_mu_h"));
        mu_z.CopyFromVec(noiseparams_reader.Value(key + "_mu_z"));
        var_z.CopyFromVec(noiseparams_reader.Value(key + "_var_z"));
      }

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG<< "Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
      }

        // compensate the mean and variance
      Vector<double> noise_mean(mean), noise_var(var);
      Matrix<double> Jx, Jz;
      CompensateDiagGaussian_FBank(mu_h, mu_z, var_z, have_energy, num_fbank,
                                   noise_mean,
                                   noise_var, Jx,
                                   Jz);

      // compute stats from mean and variance, a dummy version, the count is 1
      // the purpose is to make sure the mean and var computed from this new stats
      // are the same as noise_mean, noise_var.
      Matrix<double> noise_cmvn_stats(cmvn_stats.NumRows(),
                                      cmvn_stats.NumCols(),
                                      kSetZero);
      noise_cmvn_stats(0, feat_dim) = 1.0;
      for (int32 i = 0; i < feat_dim; ++i) {
        noise_cmvn_stats(0, i) = noise_mean(i);
        noise_cmvn_stats(1, i) = noise_var(i) + noise_mean(i) * noise_mean(i);
      }

      ApplyCmvn(noise_cmvn_stats, norm_vars, &feat);

      feat_writer.Write(key, feat);
    }
    return 0;

  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

