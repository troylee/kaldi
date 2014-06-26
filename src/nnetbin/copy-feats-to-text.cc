/*
 * copy-feat-to-textb.cc
 *
 *  Created on: Mar 22, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Write the archieve feature file to text files.
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
        "Copy features into text format.\n"
            "Usage: copy-feats-to-text [options] in-rspecifier\n";

    ParseOptions po(usage);

    std::string data_directory = "";
    po.Register("data-directory", &data_directory, "The directory for the text data.");

    std::string data_suffix = "";
    po.Register("data-suffix", &data_suffix, "The suffix for the text data");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);

    if (data_directory != "") {
      data_directory += "/";
    }

    int32 total_frames = 0;

    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {

      std::string key = kaldi_reader.Key();
      const Matrix<BaseFloat> &feat = kaldi_reader.Value();

      std::string fname = data_directory+key;
      if (data_suffix!=""){
        fname = fname + "." +data_suffix;
      }
      std::ofstream fdat(fname.c_str());

      for(int32 r=0; r<feat.NumRows(); ++r){
        for (int32 c=0; c<feat.NumCols(); ++c){
          fdat << feat(r,c) << " ";
        }
        fdat << std::endl;
      }

      fdat.close();
      total_frames += feat.NumRows();
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

