// nnetbin/est-feat-masks.cc
//
// Estimate the masks for each utterance based on the NN posteriors and
// prior mask patterns.
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
            "Usage:  est-feat-masks [options] <pat-wxfilename> <post-rspecifier>\n"
            "e.g.: \n"
            " est-feat-masks --data-directory=mask_est --data-suffix=txt mask_patterns scp:post.scp\n";

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
        pdf_rspecifier = po.GetArg(2);

    if (data_directory != "") {
      data_directory += "/";
    }

    SequentialInt32VectorReader pdf_reader(pdf_rspecifier);

    Timer tim;

    int32 num_done = 0;
    Matrix<BaseFloat> patterns;
    {
      bool binary;
      Input ki(pat_wxfilename, &binary);
      patterns.Read(ki.Stream(), binary);
    }

    for (; !pdf_reader.Done(); pdf_reader.Next()) {
      // get the keys
      std::string utt = pdf_reader.Key();
      std::vector<int32> labs = pdf_reader.Value();

      std::string fname = data_directory + utt;
      if (data_suffix != ""){
        fname = fname + "." + data_suffix;
      }
      std::ofstream fdat(fname.c_str());

      for (int32 r = 0; r < labs.size(); ++r) {
        for (int32 c = 0; c < patterns.NumCols(); ++c) {
          fdat << patterns(labs[r], c) << " ";
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
