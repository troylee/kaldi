/*
 * normalize-feats.cc
 *
 *  Created on: Sep 12, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Gaussianalize features per frame.\n"
            "Usage: normalize-feats [options] <feats-rspecifier> <outs-wspecifier>\n"
            "e.g.\n"
            "normalize-feats --norm-vars=true ark:feats.ark ark:outs.ark\n";

    ParseOptions po(usage);

    bool norm_vars = false;
    po.Register("norm-vars", &norm_vars, "Normalize variance.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats_rspecifier = po.GetArg(1),
        outs_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    BaseFloatMatrixWriter outs_writer(outs_wspecifier);

    Vector<BaseFloat> mean, var;
    int32 num_done = 0;

    for (; !feats_reader.Done(); feats_reader.Next()) {
      std::string key = feats_reader.Key();
      Matrix<BaseFloat> feats = feats_reader.Value();

      mean.Resize(feats.NumRows(), kSetZero);
      var.Resize(feats.NumRows(), kSetZero);

      mean.AddColSumMat(1.0 / feats.NumCols(), feats, 0.0);  // mean

      feats.AddVecToCols(-1.0, mean);  // subtract mean

      if (norm_vars) {
        feats.ApplyPow(2.0);
        var.AddColSumMat(1.0 / feats.NumCols(), feats, 0.0);  // variance
        var.ApplyPow(0.5);  // std var
        feats.ApplyPow(0.5);  // revert back
        var.InvertElements();
        feats.MulRowsVec(var);
      }

      outs_writer.Write(key, feats);
      ++num_done;

      if (num_done % 1000 == 0) {
        KALDI_LOG<< "Processed " << num_done << " utterances.\n";
      }
    }

    KALDI_LOG<< "Totally processed " << num_done << " utterances.\n";

  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

