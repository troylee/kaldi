// nnetbin/analyze-rec-errors.cc
//
// compute the number of different errors for the interpolation system
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Conunt different error stats for the interpolation system of two.\n"
        "Usage:  analyze-rec-errs  [options] <s1-tra-rspecifier> <s2-tra-rspecifier> "
        "<s3-tra-rspecifier> <ref-tra-rspecifier> [<s1w-s2w-s3c-wspecifier>]\n"
        "e.g.: \n"
        " analyze-rec-errs ark:s1.tra ark:s2.tra ark:s3.tra ark:ref.tra\n";
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (!(po.NumArgs() == 4 || po.NumArgs() == 5) ) {
      po.PrintUsage();
      exit(1);
    }

    std::string s1_rspecifier = po.GetArg(1), /* sub system 1 */
        s2_rspecifier = po.GetArg(2), /* sub system 2 */
        s3_rspecifier = po.GetArg(3), /* combined system of the above two */
        ref_rspecifier = po.GetArg(4), /* reference */
        out_wspecifier = po.GetOptArg(5); /* the optional output file to save the files that are wrong for two
        sub systems but correct after combination. */

    SequentialInt32VectorReader ref_reader(ref_rspecifier);
    RandomAccessInt32VectorReader s1_reader(s1_rspecifier);
    RandomAccessInt32VectorReader s2_reader(s2_rspecifier);
    RandomAccessInt32VectorReader s3_reader(s3_rspecifier);

    Int32VectorWriter out_writer(out_wspecifier);

    int32 num_done = 0, num_no_rec = 0, num_len_err = 0, num_frames = 0;
    int32 s1_err = 0, s2_err = 0, s3_err = 0;
    int32 s1w_s2c_s3c = 0, s1c_s2w_s3c = 0, s1w_s2w_s3c = 0;

    for (; !ref_reader.Done(); ref_reader.Next()) {
      std::string key = ref_reader.Key();
      std::vector<int32> labs = ref_reader.Value();

      if(!s1_reader.HasKey(key) || !s2_reader.HasKey(key) || !s3_reader.HasKey(key)){
        KALDI_WARN << "Recognition results for utterance " << key << " not found!";
        ++num_no_rec;
        continue;
      }

      std::vector<int32> s1_rec = s1_reader.Value(key);
      std::vector<int32> s2_rec = s2_reader.Value(key);
      std::vector<int32> s3_rec = s3_reader.Value(key);

      bool has_type3_err = false;
      std::vector<int32> out_rec(labs.size(), 0);

      if ((s1_rec.size()!=labs.size()) || (s2_rec.size()!=labs.size()) || (s3_rec.size()!=labs.size())){
        KALDI_WARN << "Mismatched recognition length!";
        ++num_len_err;
      }

      for (size_t i = 0; i < labs.size(); i++) {
        if(s1_rec[i]!=labs[i]) ++s1_err;
        if(s2_rec[i]!=labs[i]) ++s2_err;
        if(s3_rec[i]!=labs[i]) ++s3_err;
        if((s1_rec[i]!=labs[i]) && (s2_rec[i]==labs[i]) && (s3_rec[i]==labs[i])) ++s1w_s2c_s3c;
        if((s1_rec[i]==labs[i]) && (s2_rec[i]!=labs[i]) && (s3_rec[i]==labs[i])) ++s1c_s2w_s3c;
        if((s1_rec[i]!=labs[i]) && (s2_rec[i]!=labs[i]) && (s3_rec[i]==labs[i])) {
          ++s1w_s2w_s3c;
          out_rec[i]=1;
          has_type3_err=true;
        }
      }

      if(out_wspecifier != "" && has_type3_err){
        out_writer.Write(key, out_rec);
      }
      num_done++;
      num_frames += labs.size();
    }

    KALDI_LOG << "Processed " << num_done << " recognition results, "
        << num_no_rec << " have no recognition results,"
        << num_len_err << " have mismatched length.";
    KALDI_LOG << "###########################################################";
    KALDI_LOG << "Total number of frames: " << num_frames;
    KALDI_LOG << "S1 error count: " << s1_err << " (" << s1_err*100.0/num_frames << "%)";
    KALDI_LOG << "S2 error count: " << s2_err << " (" << s2_err*100.0/num_frames << "%)";
    KALDI_LOG << "S3 error count: " << s3_err << " (" << s3_err*100.0/num_frames << "%)";
    KALDI_LOG << "S1 wrong, S2 correct, S3 correct count: " << s1w_s2c_s3c << " (" << s1w_s2c_s3c*100.0/num_frames << "%)";
    KALDI_LOG << "S1 correct, S2 wrong, S3 correct count: " << s1c_s2w_s3c << " (" << s1c_s2w_s3c*100.0/num_frames << "%)";
    KALDI_LOG << "S1 wrong, S2 wrong, S3 correct count: " << s1w_s2w_s3c << " (" << s1w_s2w_s3c*100.0/num_frames << "%)";
    KALDI_LOG << "###########################################################";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


