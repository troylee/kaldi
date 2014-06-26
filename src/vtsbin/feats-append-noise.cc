/*
 * feats-append-noise.cc
 *
 *  Created on: Feb 20, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Append VTS noise estimation to the feature frames\n"
        "Usage: feats-append-noise [options] feat-rspecifier noise-rspecifier out-wspecifier\n"
        "Example: feats-append-noise ark:feats.ark ark:noise.ark ark:output.ark\n";

    ParseOptions po(usage);

    int32 num_cepstral = 13;

    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral coefficients");

    bool use_additive_noise_mean = true;
    bool use_additive_noise_var = false;
    bool use_channel_noise_mean = false;

    po.Register("use-additive-noise-mean", &use_additive_noise_mean,
                "Append the additive noise mean to each feature frame");
    po.Register("use-additive-noise-var", &use_additive_noise_var,
                "Append the additive noise var to each feature frame");
    po.Register("use-channel-noise-mean", &use_channel_noise_mean,
                "Append the channel noise mean to each feature frame");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1);
    std::string noise_rspecifier = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessDoubleVectorReader noise_reader(noise_rspecifier);

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      Vector<double> noise;
      int32 dim = 0;

      if (use_additive_noise_mean) {
        if (!noise_reader.HasKey(utt + "_mu_z")) {
          KALDI_WARN<< "Could not find additive noise mean for " << utt << " in "
          << noise_rspecifier << ": producing no output for the utterance";
          continue;
        }
        Vector<double> cur(noise_reader.Value(utt + "_mu_z"));
        noise.Resize(dim+num_cepstral, kCopyData); // only static part has values
        (SubVector<double>(noise, dim, num_cepstral)).CopyFromVec(SubVector<double>(cur, 0, num_cepstral));
        dim=noise.Dim();
      }

      if (use_additive_noise_var) {
        if (!noise_reader.HasKey(utt + "_var_z")) {
          KALDI_WARN<< "Could not find additive noise var for " << utt << " in "
          << noise_rspecifier << ": producing no output for the utterance";
          continue;
        }
        Vector<double> cur(noise_reader.Value(utt + "_var_z"));
        noise.Resize(dim+cur.Dim(), kCopyData); // var has all the values
        (SubVector<double>(noise, dim, cur.Dim())).CopyFromVec(cur);
        dim=noise.Dim();
      }

      if (use_channel_noise_mean) {
        if (!noise_reader.HasKey(utt + "_mu_h")) {
          KALDI_WARN<< "Could not find channel noise mean for " << utt << " in "
          << noise_rspecifier << ": producing no output for the utterance";
          continue;
        }
        Vector<double> cur(noise_reader.Value(utt + "_mu_h"));
        noise.Resize(dim+num_cepstral, kCopyData);
        (SubVector<double>(noise, dim, num_cepstral)).CopyFromVec(SubVector<double>(cur, 0, num_cepstral));
        dim=noise.Dim();
      }

      int32 num_frames = feats.NumRows();
      Matrix<BaseFloat> output_feats(num_frames, feats.NumCols()+noise.Dim());
      (SubMatrix<BaseFloat>(output_feats, 0, num_frames, 0, feats.NumCols())).CopyFromMat(feats);
      (SubMatrix<BaseFloat>(output_feats, 0, num_frames, feats.NumCols(), noise.Dim())).CopyRowsFromVec(Vector<BaseFloat>(noise));

      KALDI_VLOG(1) << "Utterance : " << utt << ": # of frames = "
          << num_frames;

      kaldi_writer.Write(utt, output_feats);

    }

    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

