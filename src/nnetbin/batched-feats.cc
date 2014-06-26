// nnetbin/batched-feats.cc
/*
 * Generate batched features and labels for use in Python.
 *
 * It expects spliced features (e.g. generated from splice-feats) and
 * corresponding labels.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Generate features with labels for use in Python. Expect feats from spliced-feats.\n"
            "Usage:  batched-feats [options] <output-dir> <feature-rspecifier> [<alignments-rspecifier>]\n"
            "e.g.: \n"
            " batched-feats output_dir ark:features.ark ark:ali.ark \n";

    ParseOptions po(usage);

    int32 batch_size = 1024;
    po.Register("batch-size", &batch_size, "Number of instances in one batch");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string output_dir = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetOptArg(3);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader;
    if (alignments_rspecifier != "") {
      alignments_reader.Open(alignments_rspecifier);
    }

    int32 num_done = 0, num_other_error = 0, batch_id = 1, cur_count = 0;
    char str[100];
    sprintf(str, "%d", batch_id);
    std::string fname = output_dir + "/data_batch_" + str;
    FILE *fp_data = fopen(fname.c_str(), "w");
    fname = output_dir + "/label_batch_" + str;
    FILE *fp_label = fopen(fname.c_str(), "w");

    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();

      if (alignments_rspecifier != "") {
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        if ((int32) alignment.size() != feats.NumRows()) {
          KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (feats.NumRows());
          num_other_error++;
          continue;
        }

        int32 dim = feats.NumCols();
        for (int32 r = 0; r < feats.NumRows(); ++r) {
          for (int32 c = 0; c < dim; ++c) {
            fprintf(fp_data, "%f ", feats(r, c));
          }
          fprintf(fp_data, "\n");
          fprintf(fp_label, "%d\n", alignment[r]);
          ++cur_count;

          if (cur_count == batch_size) {
            fclose(fp_data);
            fclose(fp_label);
            ++batch_id;
            KALDI_LOG<< "Batch: " << batch_id;
            sprintf(str, "%d", batch_id);
            std::string fname = output_dir + "/data_batch_" + str;
            fp_data = fopen(fname.c_str(), "w");
            fname = output_dir + "/label_batch_" + str;
            fp_label = fopen(fname.c_str(), "w");
            cur_count = 0;
          }
        }
      } else {
        int32 dim = feats.NumCols();
        for (int32 r = 0; r < feats.NumRows(); ++r) {
          for (int32 c = 0; c < dim; ++c) {
            fprintf(fp_data, "%f ", feats(r, c));
          }
          fprintf(fp_data, "\n");
          fprintf(fp_label, "0\n");
          ++cur_count;

          if (cur_count == batch_size) {
            fclose(fp_data);
            fclose(fp_label);
            ++batch_id;
            KALDI_LOG<< "Batch: " << batch_id;
            sprintf(str, "%d", batch_id);
            std::string fname = output_dir + "/data_batch_" + str;
            fp_data = fopen(fname.c_str(), "w");
            fname = output_dir + "/label_batch_" + str;
            fp_label = fopen(fname.c_str(), "w");
            cur_count = 0;
          }
        }
      }
      ++num_done;
      if (num_done % 100 == 0) {
        KALDI_LOG<< num_done << " done.";
      }
    }
    fclose(fp_data);
    fclose(fp_label);

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
