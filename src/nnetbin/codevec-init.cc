/*
 * codevec-init.cc
 *
 *  Created on: Oct 5, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Initialize code vectors.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]){
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try{

    const char *usage =
        "Initialize code vectors to zeros for each set\n"
        "Usage: codevec-init [options] <set2utt-map> <codevec-wspecifier> "
        "e.g.: \n"
        " codevec-init --code-dim=10 ark:set2utt.map ark,t:code.ark \n";

    ParseOptions po(usage);

    int32 code_dim = 0;
    po.Register("code-dim", &code_dim, "Dimension of the code vector");

    bool add_gauss_noise = false;
    po.Register("add_gauss-noise", &add_gauss_noise, "Whether to add Gaussian noise to the code vector");

    po.Read(argc, argv);

    if (po.NumArgs() != 2){
      po.PrintUsage();
      exit(1);
    }

    std::string set2utt_filename = po.GetArg(1),
        codevec_wspecifier = po.GetArg(2);

    SequentialTokenVectorReader set2utt_reader(set2utt_filename);
    BaseFloatVectorWriter codevec_writer(codevec_wspecifier);

    Vector<BaseFloat> codevec(code_dim, kSetZero);

    int32 num_set = 0;
    for ( ; !set2utt_reader.Done(); set2utt_reader.Next()) {
      std::string key = set2utt_reader.Key();

      if(add_gauss_noise){
        codevec.SetRandn();
      }

      codevec_writer.Write(key, codevec);
      ++num_set;
    }

    KALDI_LOG << "Created " << num_set << (add_gauss_noise?" random ":"") << " code vectors.";

  }catch (const std::exception &e){
    std::cerr<< e.what() << '\n';
    return -1;
  }
}




