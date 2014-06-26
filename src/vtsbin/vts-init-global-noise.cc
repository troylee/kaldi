/*
 * vts-init-global-noise.cc
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
 *  Created on: Sept 4, 2013
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
        "Estimate the initial global noise parameters.\n"
            "Usage: vts-init-global-noise [options] features-rspecifier noiseparams-wspecifier\n";
    ParseOptions po(usage);
    int32 num_static = 13;
    int32 noise_frames = 20;
    bool zero_mu_z_deltas = true;

    po.Register("num-static", &num_static,
                "Dimension of the static feature coefficients");
    po.Register("noise-frames", &noise_frames,
                "Number of frames at the beginning and ending of"
                " each sentence used for noise estimation");

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

    int32 feat_dim = 3 * num_static;
    // noise model parameters, current estimate and last estimate
    Vector<double> mu_h(feat_dim), mu_z(feat_dim), var_z(feat_dim);
    mu_h.SetZero();  // the convolutional noise, initialized to be 0
    mu_z.SetZero();  // the additive noise mean, only static part to be estimated
    var_z.SetZero();  // the additive noise covariance, diagonal, to be estimated

    Vector<double> mean_obs(feat_dim);  // mean of
    Vector<double> x2(feat_dim);  // x^2 stats
    int32 i, tot_frames;

    int32 num_fail = 0, num_success = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();

      KALDI_VLOG(1)<< "Current utterance: " << key;

      if (features.NumRows() == 0) {
        KALDI_WARN<< "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      if (features.NumRows() < noise_frames * 2) {
        KALDI_WARN<< "Too short utterance for VTS: " << key
            << " (" << features.NumRows() << " frames)";
      }

      if (features.NumCols() != 3 * num_static) {
        KALDI_WARN<< "Current feature dim is not " << 3 * num_static <<" !";
      }

        // accumulate the starting stats
      for (i = 0; i < noise_frames && i < features.NumRows(); ++i) {
        mu_z.AddVec(1.0, features.Row(i));
        var_z.AddVec2(1.0, features.Row(i));
      }
      tot_frames += i;
      // accumulate the ending stats
      i = features.NumRows() - noise_frames;
      if (i < 0) i = 0;
      for (; i < features.NumRows(); ++i, ++tot_frames) {
        mu_z.AddVec(1.0, features.Row(i));
        var_z.AddVec2(1.0, features.Row(i));
      }

      ++num_success;
    }

    mu_z.Scale(1.0 / tot_frames);
    var_z.Scale(1.0 / tot_frames);
    var_z.AddVec2(-1.0, mu_z);

    ///////////////////////////////////////////////////
    // As we assume the additive noise has 0 delta and delta-delta mean
    if (zero_mu_z_deltas) {
      for (i = num_static; i < feat_dim; ++i) {
        mu_z(i) = 0.0;
      }
    }

    // Writting the noise parameters out
    noiseparams_writer.Write("global_mu_h", mu_h);
    noiseparams_writer.Write("global_mu_z", mu_z);
    noiseparams_writer.Write("global_var_z", var_z);

    KALDI_LOG<< "Done " << num_success << " utterances successfully, "
        << num_fail << " failed.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

