/*
 * compute-average-noise.cc
 *
 * Load in the VTS estimated noise parameters per utterance,
 * output the average noise parameters. Simple average.
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
        "Compute the average mean and covariance.\n"
        "Usage: compute-average-noise [options] features-rspecifier "
        "per-utt-noise-rspecifier global-noise-wspecifier\n";
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
        perutt_rspecifier = po.GetArg(2),
        global_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(perutt_rspecifier);
    DoubleVectorWriter global_writer(global_wspecifier);

    // only the additive noise mean and variance are estimation
    // convolutional noises are set 0
    Vector<double> global_mu_z(39, kSetZero), global_mu_h(39, kSetZero),
        global_var_z(39, kSetZero);
    std::vector < std::string > feature_keys;

    int32 num_fail = 0, num_success = 0, tot_counts = 0;
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

      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR
            << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
      }

      int32 feat_dim = features.NumCols();
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

      global_mu_h.AddVec(1.0, mu_h);
      global_mu_z.AddVec(1.0, mu_z);
      global_var_z.AddVec(1.0, var_z);
      tot_counts += 1;
      feature_keys.push_back(key);

      ++num_success;
    }

    // compute the global noise mean and variance
    global_mu_h.Scale(1.0 / tot_counts);
    global_mu_z.Scale(1.0 / tot_counts);
    global_var_z.Scale(1.0 / tot_counts);

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

