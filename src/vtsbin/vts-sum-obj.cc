/*
 * vtsbin/vts-sum-obj.cc
 *
 *  Created on: Nov 25, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Sum all the subset objective values.
 *
 */

#include "util/common-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for objective function evaluation.\n"
            "Usage: vts-sum-obj [options] stats-out stats-in1 stats-in2 ...\n";

    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    BaseFloat tot_like = 0.0;
    int32 tot_file = 0, tot_frame = 0;

    std::string obj_wxfilename = po.GetArg(1);

    int num_accs = po.NumArgs() - 1;
    for (int i = 2, max = po.NumArgs(); i <= max; i++) {
      BaseFloat cur_like = 0.0;
      int32 cur_file = 0, cur_frame = 0;

      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      Input ki(stats_in_filename, &binary_read);
      ExpectToken(ki.Stream(), binary_read, "<log_likelihood>");
      ReadBasicType(ki.Stream(), binary_read, &cur_like);
      ExpectToken(ki.Stream(), binary_read, "<num_file>");
      ReadBasicType(ki.Stream(), binary_read, &cur_file);
      ExpectToken(ki.Stream(), binary_read, "<num_frame>");
      ReadBasicType(ki.Stream(), binary_read, &cur_frame);

      tot_like += cur_like;
      tot_file += cur_file;
      tot_frame += cur_frame;
    }

    KALDI_LOG<< "Summed " << num_accs << " stats.\n Total "
        << tot_file << " files, avg log like/file "
        << (tot_like / tot_file) << "\n Total "
        << tot_frame << " frames, avg log like/frame "
        << (tot_like / tot_frame);

    // Write out the accs
    {
      Output ko(obj_wxfilename, binary);
      WriteToken(ko.Stream(), binary, "<log_likelihood>");
      WriteBasicType(ko.Stream(), binary, tot_like);
      WriteToken(ko.Stream(), binary, "<num_file>");
      WriteBasicType(ko.Stream(), binary, tot_file);
      WriteToken(ko.Stream(), binary, "<num_frame>");
      WriteBasicType(ko.Stream(), binary, tot_frame);
    }
    KALDI_LOG<< "Written stats to " << obj_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

