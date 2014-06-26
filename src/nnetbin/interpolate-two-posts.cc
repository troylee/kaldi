/*
 * nnetbin/interpolate-two-posts.cc
 *
 *  Created on: Jun 18, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Interpolate two sets of posteriors.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Interpolate two sets of posteriors frame by frame.\n"
            "Usage: interpolate-two-posts [options] posts1-rspecifier posts2-rspecifier "
            "out-posts-wspecifier\n";

    ParseOptions po(usage);
    BaseFloat posts1_scale = 1.0;
    po.Register("posts1-scale", &posts1_scale, "Interpolation weight for posterior 1");

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Apply log on the final output");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string posts1_rspecifier = po.GetArg(1);
    std::string posts2_rspecifier = po.GetArg(2);
    std::string out_posts_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader posts1_reader(posts1_rspecifier);
    // to save the computational cost, we assume the two features are in the same key order
    //RandomAccessBaseFloatMatrixReader feats2_reader(feats2_rspecifier);
    SequentialBaseFloatMatrixReader posts2_reader(posts2_rspecifier);
    BaseFloatMatrixWriter out_posts_writer(out_posts_wspecifier);

    int32 num_done = 0;

    for (; !posts1_reader.Done() && !posts2_reader.Done(); posts1_reader.Next(), posts2_reader.Next()) {
      std::string key = posts1_reader.Key();
      Matrix<BaseFloat> posts1(posts1_reader.Value());

      if (posts2_reader.Key() != key) {
        KALDI_ERR<< "Key mismtach for the two features, "
        << "make sure they have the same ordering.";
      }

      Matrix<BaseFloat> posts2(posts2_reader.Value());

      KALDI_ASSERT(posts1.NumRows()==posts2.NumRows() && posts1.NumCols() == posts2.NumCols());

      posts1.Scale(posts1_scale);
      posts1.AddMat(1-posts1_scale, posts2, kNoTrans);

      if (apply_log){
        posts1.ApplyLog();
      }

      out_posts_writer.Write(key, posts1);

      ++num_done;
      if (num_done % 1000 == 0) {
        KALDI_LOG<< "Done " << num_done << " utterances.";
      }
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

