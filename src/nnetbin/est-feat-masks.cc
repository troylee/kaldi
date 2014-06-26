// nnetbin/est-feat-masks-with-pdf.cc
//
// Estimate the masks for each utterance based on the pdf label for each frame.
// i.e. simply selecting the patterns.
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Estimate masks for each utterance and save to the specific folder.\n"
            "Usage:  est-feat-masks-with-pdf [options] <pat-wxfilename> <pdf-rspecifier>\n"
            "e.g.: \n"
            " est-feat-masks-with-pdf --data-directory=mask_est --data-suffix=txt mask_patterns scp:pdf.scp\n";

    ParseOptions po(usage);

    std::string data_directory = "";
    po.Register("data-directory", &data_directory,
                "The directory for the text data.");

    std::string data_suffix = "";
    po.Register("data-suffix", &data_suffix, "The suffix for the text data");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string pat_wxfilename = po.GetArg(1),
        post_rspecifier = po.GetArg(2);

    if (data_directory != "") {
      data_directory += "/";
    }

    SequentialBaseFloatMatrixReader post_reader(post_rspecifier);

    Timer tim;

    int32 num_done = 0;
    Matrix<BaseFloat> patterns;
    {
      bool binary;
      Input ki(pat_wxfilename, &binary);
      patterns.Read(ki.Stream(), binary);
    }

    for (; !post_reader.Done(); post_reader.Next()) {
      // get the keys
      std::string utt = post_reader.Key();
      const Matrix<BaseFloat> &post = post_reader.Value();

      Matrix<BaseFloat> mask(post.NumRows(), patterns.NumCols());
      mask.AddMatMat(1.0, post, kNoTrans, patterns, kNoTrans, 0.0);

      std::ofstream fdat((data_directory + utt + "." + data_suffix).c_str());

      for (int32 r = 0; r < mask.NumRows(); ++r) {
        for (int32 c = 0; c < mask.NumCols(); ++c) {
          fdat << mask(r, c) << " ";
        }
        fdat << std::endl;
      }

      fdat.close();

      num_done++;
      if (num_done % 1000 == 0) {
        KALDI_LOG<< "Done " << num_done << " files.";
      }

    }

    std::cout << "\n" << std::flush;
    KALDI_LOG<< "COMPUTATION" << " FINISHED " << tim.Elapsed() << "s";
    KALDI_LOG<< "Done " << num_done << " files.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
