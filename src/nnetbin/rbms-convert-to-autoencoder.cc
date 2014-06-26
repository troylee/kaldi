// gmmbin/rbm-convert-to-nnet.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "nnet/nnet-rbm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Combining RBMs sequentially to an Auto-Encoder\n"
            "Usage:  rbms-convert-to-autoencoder [options] <ae-out> <rbm-in1> <rbm-in2> ... \n"
            "e.g.:\n"
            " rbms-convert-to-autoencoder --binary=false ae.mdl rbm1.mdl rbm2.mdl ...\n";

    bool binary_write = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    int total_args = po.NumArgs();
    KALDI_LOG << total_args << " arguments";

    std::string model_out_filename = po.GetArg(1);

    {
      Output ko(model_out_filename, binary_write);

      // Encoder
      for (int i = 2; i <= total_args; ++i) {

        KALDI_LOG << "Encoder: " << i;

        std::string model_in_filename = po.GetArg(i);

        Nnet nnet;
        {
          bool binary_read;
          Input ki(model_in_filename, &binary_read);
          nnet.Read(ki.Stream(), binary_read);
        }

        KALDI_ASSERT(nnet.LayerCount() == 1);
        KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRbm);
        RbmBase& rbm = dynamic_cast<RbmBase&>(*nnet.Layer(0));

        rbm.WriteAsAutoEncoder(ko.Stream(), true, binary_write);

      }

      // Decoder
      for (int i = total_args; i >= 2 ; --i) {

        KALDI_LOG << "Decoder: " << i;

        std::string model_in_filename = po.GetArg(i);

        Nnet nnet;
        {
          bool binary_read;
          Input ki(model_in_filename, &binary_read);
          nnet.Read(ki.Stream(), binary_read);
        }

        KALDI_ASSERT(nnet.LayerCount() == 1);
        KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRbm);
        RbmBase& rbm = dynamic_cast<RbmBase&>(*nnet.Layer(0));

        rbm.WriteAsAutoEncoder(ko.Stream(), false, binary_write);

      }

    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

