/*
 * compute-utt-stats.cc
 *
 * Generate the initial noise parameters assuming the whole utterance is noise, including:
 *  mu_h - convolutional noise mean;
 *  mu_z - additive noise mean;
 *  var_z - additive noise variance.
 *
 *  The estimation is using the first and last frames,
 *  which are assumued to be background noise only.
 *
 *  We also assume the noise are independent of state
 *  and thus only having static coefficients.
 *  The variance is assumed diagonal with both static
 *  and dynamic parameters.
 *
 *  Created on: Oct 22, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char* argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Compute the mean and covariance per utterance and through out the whole bunch of features.\n"
        "Usage: compute-utt-noise [options] features-rspecifier per-utt-stats-wspecifier global-stats-wspecifier\n";
    ParseOptions po(usage);
    int32 num_static = 13;

    po.Register("num-static", &num_static,
                "Dimension of the static feature coefficients");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        perutt_wspecifier = po.GetArg(2),
        global_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    DoubleVectorWriter perutt_writer(perutt_wspecifier);
    DoubleVectorWriter global_writer(global_wspecifier);

    // only the additive noise mean and variance are estimation
    // convolutional noises are set 0
    Vector<double> global_mu_z(39, kSetZero), global_mu_h(39, kSetZero),
        global_var_z(39, kSetZero);
    std::vector < std::string > feature_keys;

    int32 num_fail = 0, num_success = 0, tot_frames = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Current utterance: " << key;
      }

      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      int32 feat_dim = features.NumCols();
      if (feat_dim != 39) {
        KALDI_ERR
            << "Could not decode the features, only 39D MFCC_0_D_A is supported!";
      }

      // noise model parameters, current estimate and last estimate
      Vector<double> mu_z(feat_dim, kSetZero), var_z(feat_dim, kSetZero);

      for (int32 i = 0; i < features.NumRows(); ++i) {
        mu_z.AddVec(1.0, features.Row(i));
        var_z.AddVec2(1.0, features.Row(i));
      }

      global_mu_z.AddVec(1.0, mu_z);
      global_var_z.AddVec(1.0, var_z);
      tot_frames += features.NumRows();
      feature_keys.push_back(key);

      mu_z.Scale(1.0 / features.NumRows());
      var_z.Scale(1.0 / features.NumRows());
      var_z.AddVec2(-1.0, mu_z);

      // keep only the static part
      for (int32 i = num_static; i < features.NumCols(); ++i) {
        mu_z(i) = 0.0;
        var_z(i) = 0.0;
      }

      // Writting the noise parameters out
      perutt_writer.Write(key + "_mu_h", global_mu_h);
      perutt_writer.Write(key + "_mu_z", mu_z);
      perutt_writer.Write(key + "_var_z", var_z);

      ++num_success;
    }

    // compute the global noise mean and variance
    global_mu_z.Scale(1.0 / tot_frames);
    global_var_z.Scale(1.0 / tot_frames);
    global_var_z.AddVec2(-1.0, global_mu_z);
    // keep only the static part
    for (int32 i = num_static; i < 39; ++i) {
      global_mu_z(i) = 0.0;
      global_var_z(i) = 0.0;
    }

    for (int32 i = 0; i < feature_keys.size(); ++i) {
      std::string key = feature_keys[i];

      global_writer.Write(key + "_mu_h", global_mu_h);
      global_writer.Write(key + "_mu_z", global_mu_z);
      global_writer.Write(key + "_var_z", global_var_z);
    }

    KALDI_LOG << "Done " << num_success << " utterances successfully, "
        << num_fail << " failed.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

