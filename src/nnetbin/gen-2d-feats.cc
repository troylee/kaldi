// nnetbin/gen-2d-feats.cc

/*
 * Extract the first 2-dimensional features of a specified alignment PDF
 * according to the given PDF alignment.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Generate 2D features\n"
            "Usage: gen-2d-feats [options] in-feats-rspecifier pdf-align-rspecifier out-feats-wspecifier\n";

    ParseOptions po(usage);

    int32 feat_dim = 2, target_pdf = 0;
    po.Register("feat-dim", &feat_dim,
                "Number of first N dimensions to output.");
    po.Register("target-pdf", &target_pdf,
                "Target PDF ID to select the features, starts from 0.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats_rspecifier = po.GetArg(1),
        pdf_rspecifier = po.GetArg(2),
        feats_wspecifier = po.GetArg(3);

    KALDI_ASSERT(feat_dim > 0);
    KALDI_ASSERT(target_pdf >= 0);

    BaseFloatMatrixWriter feats_writer(feats_wspecifier);

    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    RandomAccessInt32VectorReader pdf_reader(pdf_rspecifier);

    int32 tot_frames = 0;
    Matrix<BaseFloat> mats;

    for (; !feats_reader.Done(); feats_reader.Next()) {

      std::string key = feats_reader.Key();
      const Matrix<BaseFloat> &feats = feats_reader.Value();

      const std::vector<int32> &pdfs = pdf_reader.Value(key);
      // 1st iteration to count the number of frames
      tot_frames = 0;
      for (int32 i = 0; i < pdfs.size(); ++i) {
        if (pdfs[i] == target_pdf)
          ++tot_frames;
      }

      if (tot_frames > 0) {
        mats.Resize(tot_frames, feat_dim, kSetZero);
        for (int32 i = 0, k = 0; i < pdfs.size(); ++i) {
          if (pdfs[i] == target_pdf) {
            (mats.Row(k)).CopyFromVec(
                SubVector<BaseFloat>(feats.Row(i), 0, feat_dim));
            ++k;
          }
        }

        feats_writer.Write(key, mats);
      }
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

