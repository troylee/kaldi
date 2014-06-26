/*
 * copy-feat-from-textb.cc
 *
 *  Created on: Mar 22, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Write the archive feature file from text files.
 *
 */

#include <fstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy features from text format.\n"
            "Usage: copy-feats-from-text [options] fname-list out-wspecifier\n";

    ParseOptions po(usage);

    int32 dim = 0;
    po.Register("dim", &dim, "Feature dimension");

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

    std::string in_file_list = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(dim > 0);

    int32 total_files = 0, total_frames = 0;
    int32 max_frames = 1000, delta_frames = 500;

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    Matrix<BaseFloat> data(max_frames, dim, kSetZero);

    std::ifstream fscp(in_file_list.c_str());

    while (fscp.good()) {
      std::string key;
      fscp >> key;

      if(key=="") continue;

      std::string fname="";
      if (data_directory != "") {
        fname = data_directory + "/";
      }

	  fname = fname + key;

	  if (data_suffix != "") {
      	fname = fname + "." + data_suffix;
	  }

      std::ifstream fdat(fname.c_str());

      int32 num_frames = 0;
      while (fdat.good()) {
        if (num_frames >= max_frames) {
          max_frames += delta_frames;
          data.Resize(max_frames, dim, kCopyData);
        }
        for (int32 i = 0; i < dim; ++i) {
          BaseFloat val;
          fdat >> val;
          data(num_frames, i) = val;
        }
        if (fdat.good()) {
          ++num_frames;
        }
      }

      {
        Matrix<BaseFloat> feat(SubMatrix<BaseFloat>(data, 0, num_frames, 0, dim), kNoTrans);
        kaldi_writer.Write(key, feat);
      }

      total_frames += num_frames;
      total_files += 1;

      //KALDI_LOG<< "Done " << key << ", " << num_frames << " frames.";
    }

    KALDI_LOG<< "Totally " << total_files << " files, " << total_frames << " frames.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

