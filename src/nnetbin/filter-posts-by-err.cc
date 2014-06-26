// nnetbin/filter-posts-by-err.cc
//
// Based on the error patterns to filter out the posteriors of the
// two sub-systems.
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Filtering the posteriors of two subsystems' in the interpolation method.\n"
            "Usage:  filter-posts-by-err  [options] <s1-tra-rspecifier> <post1-rspecifier>"
            " <s2-tra-rspecifier> <post2-rspecifier> <s3-tra-rspecifier> <ref-tra-rspecifier>"
            " <post1-wspecifier> <post2-wspecifier>  e.g.: \n"
            " filter-posts-by-err ark:s1.tra ark:post1.ark ark:s2.tra ark:post2.ark "
            " ark:s3.tra ark:ref.tra ark:post1_filtered.ark ark:post2_filtered.ark\n";
    ParseOptions po(usage);

    int32 err_pattern = 0;
    po.Register(
                "err-pattern",
                &err_pattern,
                "Error pattern to filter out: \n"
                "[0 - at least one error in [s1, s2], s3 correct;\n"
                " 1 - s1 and s2 both wrong, s3 correct;\n"
                " 2 - s1 and s2 both correct, s3 correct].");

    po.Read(argc, argv);

    if (po.NumArgs() != 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string tra1_rspecifier = po.GetArg(1), /* sub system 1, per-frame phone label */
    post1_rspecifier = po.GetArg(2), /* sub system 1, posteriors */
    tra2_rspecifier = po.GetArg(3), /* sub system 2, per-frame phone label */
    post2_rspecifier = po.GetArg(4), /* sub system 2, posteriors */
    tra3_rspecifier = po.GetArg(5), /* the combined system per-frame phone label */
    ref_rspecifier = po.GetArg(6), /* reference */
    post1_wspecifier = po.GetArg(7), /* filtered sub system 1 posteriors */
    post2_wspecifier = po.GetArg(8); /* filtered sub system 2 posteriors */

    SequentialInt32VectorReader ref_reader(ref_rspecifier);
    RandomAccessInt32VectorReader tra1_reader(tra1_rspecifier);
    RandomAccessInt32VectorReader tra2_reader(tra2_rspecifier);
    RandomAccessInt32VectorReader tra3_reader(tra3_rspecifier);

    RandomAccessBaseFloatMatrixReader post1_reader(post1_rspecifier);
    RandomAccessBaseFloatMatrixReader post2_reader(post2_rspecifier);

    BaseFloatMatrixWriter post1_writer(post1_wspecifier);
    BaseFloatMatrixWriter post2_writer(post2_wspecifier);

    int32 num_done = 0, num_no_rec = 0, num_len_err = 0, num_frames = 0, tot_err_frames = 0;

    for (; !ref_reader.Done(); ref_reader.Next()) {
      std::string key = ref_reader.Key();
      std::vector<int32> labs = ref_reader.Value();

      if (!tra1_reader.HasKey(key) || !tra2_reader.HasKey(key)
          || !tra3_reader.HasKey(key)
          || !post1_reader.HasKey(key) || !post2_reader.HasKey(key)) {
        KALDI_WARN<< "Recognition results/posteriors for utterance " << key << " not found!";
        ++num_no_rec;
        continue;
      }

      std::vector<int32> tra1_rec = tra1_reader.Value(key);
      std::vector<int32> tra2_rec = tra2_reader.Value(key);
      std::vector<int32> tra3_rec = tra3_reader.Value(key);

      const Matrix<BaseFloat> post1 = post1_reader.Value(key);
      const Matrix<BaseFloat> post2 = post2_reader.Value(key);

      if ((tra1_rec.size() != labs.size()) || (tra2_rec.size() != labs.size())
          || (tra3_rec.size() != labs.size())
          || post1.NumRows() != labs.size() || post2.NumRows() != labs.size()
          || post1.NumCols() != post2.NumCols()) {
        KALDI_WARN<< "Dimension mismatch error!";
        ++num_len_err;
      }

      std::vector<int32> err_frames(labs.size(), 0);
      int32 num_err_frames = 0;
      for (size_t i = 0; i < labs.size(); i++) {
        switch (err_pattern) {
          case 0:
            /* at least one error in S1 and S2, S3 correct */
            if (((tra1_rec[i] != labs[i]) || (tra2_rec[i] != labs[i]))
                && (tra3_rec[i] == labs[i])) {
              err_frames[i] = 1;
              ++num_err_frames;
            }
            break;
          case 1:
            /* S1 and S2 are all wrong, S3 correct */
            if ((tra1_rec[i] != labs[i]) && (tra2_rec[i] != labs[i])
                && (tra3_rec[i] == labs[i])) {
              err_frames[i] = 1;
              ++num_err_frames;
            }
            break;

          case 2:
            /* S1 and S2 are all correct, S3 correct */
            if ((tra1_rec[i] == labs[i]) && (tra2_rec[i] == labs[i])
                && (tra3_rec[i] == labs[i])) {
              err_frames[i] = 1;
              ++num_err_frames;
            }
            break;
        }
      }

      if (num_err_frames>0){
        Matrix<BaseFloat> out_post1(num_err_frames, post1.NumCols());
        Matrix<BaseFloat> out_post2(num_err_frames, post2.NumCols());
        int32 k = 0;
        for(size_t r=0; r<err_frames.size(); ++r){
          if (err_frames[r]==1){
            out_post1.CopyRowFromVec(post1.Row(r), k);
            out_post2.CopyRowFromVec(post2.Row(r), k);
            ++k;
          }
        }

        post1_writer.Write(key, out_post1);
        post2_writer.Write(key, out_post2);
      }

      num_done++;
      num_frames += labs.size();
      tot_err_frames += num_err_frames;
    }

    KALDI_LOG<< "Processed " << num_done << " recognition results, "
    << num_no_rec << " have no recognition results,"
    << num_len_err << " have mismatched length.";
    KALDI_LOG<< "Total number of frames: " << num_frames << ", "
        << tot_err_frames << " are filtered out.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

