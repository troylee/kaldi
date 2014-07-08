// nnetbin/nnet-rm-dropout.cc

// Copyright 2014  Bo Li (li-bo@outlook.com)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Remove dropout layers from a neural network\n"
        "Usage:  nnet-rm-dropout [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " nnet-rm-dropout nnet_ori.mdl nnet_new.mdl\n";

    ParseOptions po(usage);

    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // load in the original network
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    {
      Output ko(model_out_filename, binary_write);

      for (int32 i=0; i<nnet.LayerCount(); ++i){
        Component *layer = nnet.Layer(i);
        if(layer->GetType()==Component::kDropout){
          Dropout *dp=dynamic_cast<Dropout*>(layer);
          Scale *sc=new Scale(dp->InputDim(), dp->OutputDim(), NULL);
          sc->SetScale(1- dp->GetDropRatio());
          sc->Write(ko.Stream(), binary_write);
        } else {
          layer->Write(ko.Stream(), binary_write);
        }
      }
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


