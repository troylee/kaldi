/*
 * hmmbl-create.cc
 *
 *  Created on: Sep 12, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Create a <hmmbl> layer from an HMM model.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert an existing HMM model to a <hmmbl> Nnet layer.\n"
        "Usage: hmmbl-create [option] <hmm-model-in> <nnet-layer-out>\n"
        "e.g.:\n"
        "hmmbl-create hmm nnet\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write the output in binary mode.");

    po.Read(argc, argv);

    if (po.NumArgs()!=2){
      po.PrintUsage();
      exit(1);
    }

    std::string hmm_model_in = po.GetArg(1),
        nnet_layer_out = po.GetArg(2);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(hmm_model_in, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    {
      Output ko(nnet_layer_out, binary);

      WriteToken(ko.Stream(), binary, "<hmmbl>");
      WriteBasicType(ko.Stream(), binary, am_gmm.NumGauss());
      WriteBasicType(ko.Stream(), binary, am_gmm.Dim()*2);
      if(!binary)
        (ko.Stream())<< '\n';
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Write the model to " << nnet_layer_out << ".\n";


  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


