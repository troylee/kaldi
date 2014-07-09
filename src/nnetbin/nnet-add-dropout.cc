// nnetbin/nnet-add-dropout.cc

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
        "Add dropout layers to a neural network\n"
        "Usage:  nnet-add-dropout [options] <model-in> <model-out> <layer-ids...>\n"
        "e.g.:\n"
        " nnet-add-dropout --add-to-input=true nnet_ori.mdl nnet_new.mdl 0 1 2\n";

    ParseOptions po(usage);

    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");

    bool add_to_input = false;
    po.Register("add-to-input", &add_to_input, "Whether add dropout to inputs");

    BaseFloat input_drop_ratio = 0.2;
    po.Register("input-drop-ratio", &input_drop_ratio, "Input feature drop ratio (0~1, exclusively)");

    BaseFloat hidden_drop_ratio = 0.5;
    po.Register("hidden-drop-ratio", &hidden_drop_ratio, "Hidden activation drop ratio (0~1, exclusive)");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
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

    // find out the layers where dropout to be appended
    std::vector<bool> flag(nnet.LayerCount(), false);
    for (int32 i=3; i<=po.NumArgs(); ++i) {
      std::string str=po.GetArg(i);
      flag[atoi(str.c_str())]=true;
    }

    {
      Output ko(model_out_filename, binary_write);

      // whether add dropout to input
      if(add_to_input){
        Dropout *dp=new Dropout(nnet.InputDim(), nnet.InputDim(), NULL);
        dp->SetDropRatio(input_drop_ratio);
        dp->Write(ko.Stream(), binary_write);
        KALDI_VLOG(2) << "<dropout> layer added to the input.";
      }

      for (int32 i=0; i<nnet.LayerCount(); ++i){
        (nnet.Layer(i))->Write(ko.Stream(), binary_write);
        if(flag[i]){
          Dropout *dp=new Dropout((nnet.Layer(i))->InputDim(), (nnet.Layer(i))->InputDim(), NULL);
          dp->SetDropRatio(hidden_drop_ratio);
          dp->Write(ko.Stream(), binary_write);
          KALDI_VLOG(2) << "<dropout> layer added to layer " << i;
        }
      }
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


