/*
 * init-noise-params.cc
 *
 * Generate the initial noise parameters, including:
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
#include "vts/vts-first-order.h"

int main(int argc, char* argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Estimate the initial noise parameters per utterance.\n"
            "Usage: vts-init-noise [options] features-rspecifier noiseparams-wspecifier\n";
    ParseOptions po(usage);
    int32 num_static = 13;
    int32 noise_frames = 20;
    bool zero_mu_z_deltas = true;

    po.Register("num-static", &num_static,
                "Dimension of the static feature coefficients");
    po.Register("noise-frames", &noise_frames,
                "Number of frames at the beginning and ending "
                "of each sentence used for noise estimation");

    po.Register("zero-mu-z-deltas", &zero_mu_z_deltas,
                "Constrain the deltas of additive noise to be 0s.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        noiseparams_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    DoubleVectorWriter noiseparams_writer(noiseparams_wspecifier);

    int32 num_fail = 0, num_success = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();

      KALDI_VLOG(1) << "Current utterance: " << key;

      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      if (features.NumRows() < noise_frames * 2) {
        KALDI_WARN << "Too short utterance for VTS: " << key << " ("
            << features.NumRows() << " frames)";
      }

      int32 feat_dim = features.NumCols();
      if (feat_dim != 3 * num_static) {
        KALDI_WARN << "Current feature dim is not " << 3 * num_static << " !";
      }

      // noise model parameters, current estimate and last estimate
      Vector<double> mu_h(feat_dim), mu_z(feat_dim), var_z(feat_dim);

      EstimateInitialNoiseModel(features, feat_dim, num_static, noise_frames,
                                zero_mu_z_deltas, &mu_h, &mu_z, &var_z);

      // Writting the noise parameters out
      noiseparams_writer.Write(key + "_mu_h", mu_h);
      noiseparams_writer.Write(key + "_mu_z", mu_z);
      noiseparams_writer.Write(key + "_var_z", var_z);

      ++num_success;
    }

    KALDI_LOG << "Done " << num_success << " utterances successfully, "
        << num_fail << " failed.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

