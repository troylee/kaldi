/*
 * nnet-xent-mse-split.cc
 *
 *  Created on: Sep 16, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Split the joint Xent-Mse Nnet to two separate ones.
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-biasedlinearity.h"

int main(int argc, char* argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Split the double objective Nnet to two separate ones.\n"
            "Usage: nnet-xent-mse-split <ori-nnet-filename>"
            "<out-xent-filename> <out-mse-filename>\n"
            "e.g. :\n"
            "nnet-xent-mse-split ori.nnet xent.nnet mse.nnet\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write the outputs in binary mode.");

    int32 xent_dim = 0;
    po.Register("xent-dim", &xent_dim, "Number of Xent units.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ori_nnet_filename = po.GetArg(1),
        out_xent_filename = po.GetArg(2),
        out_mse_filename = po.GetArg(3);

    Nnet nnet;
    nnet.Read(ori_nnet_filename);

    if (xent_dim < 0 || xent_dim > nnet.OutputDim()) {
      KALDI_ERR<< "Incorrect Xent dimension: " << xent_dim << ", should be in the range: [0, " << nnet.OutputDim() << "].\n";
    }

      /*
       * Split the last layer
       */
    BiasedLinearity &bl = dynamic_cast<BiasedLinearity&>(*(nnet.Layer(
        nnet.LayerCount() - 1)));

    Matrix<BaseFloat> weight;
    Vector<BaseFloat> bias;
    (bl.GetLinearityWeight()).CopyToMat(&weight);
    (bl.GetBiasWeight()).CopyToVec(&bias);

    int32 dim_in = bl.InputDim();
    int32 dim_mse = bl.OutputDim() - xent_dim;
    if (xent_dim > 0) {
      Output ko(out_xent_filename, binary);
      // write the shared layers first
      for (int32 i = 0; i < nnet.LayerCount() - 1; ++i) {
        (nnet.Layer(i))->Write(ko.Stream(), binary);
      }
      // write the Xent layer
      WriteToken(ko.Stream(), binary, Component::TypeToMarker(bl.GetType()));
      WriteBasicType(ko.Stream(), binary, xent_dim);
      WriteBasicType(ko.Stream(), binary, dim_in);
      if (!binary)
        (ko.Stream()) << "\n";
      (SubMatrix<BaseFloat>(weight, 0, xent_dim, 0, dim_in)).Write(ko.Stream(),
                                                                   binary);
      (SubVector<BaseFloat>(bias, 0, xent_dim)).Write(ko.Stream(), binary);
      WriteToken(ko.Stream(), binary, "<softmax>");
      WriteBasicType(ko.Stream(), binary, xent_dim);
      WriteBasicType(ko.Stream(), binary, xent_dim);
      if (!binary)
        (ko.Stream()) << "\n";

      KALDI_LOG<< "Write Xent layer to " << out_xent_filename << ".\n";
    } else {
      KALDI_WARN << "No Xent layer to be generated!";
    }

    if (dim_mse > 0) {
      Output ko(out_mse_filename, binary);
      // write shared layers first
      for (int32 i = 0; i < nnet.LayerCount() - 1; ++i) {
        (nnet.Layer(i))->Write(ko.Stream(), binary);
      }
      // write Mse layers
      WriteToken(ko.Stream(), binary, Component::TypeToMarker(bl.GetType()));
      WriteBasicType(ko.Stream(), binary, dim_mse);
      WriteBasicType(ko.Stream(), binary, dim_in);
      if (!binary)
        (ko.Stream()) << "\n";
      (SubMatrix<BaseFloat>(weight, xent_dim, dim_mse, 0, dim_in)).Write(
          ko.Stream(), binary);
      (SubVector<BaseFloat>(bias, xent_dim, dim_mse)).Write(ko.Stream(),
                                                            binary);
      WriteToken(ko.Stream(), binary, "<sigmoid>");
      WriteBasicType(ko.Stream(), binary, dim_mse);
      WriteBasicType(ko.Stream(), binary, dim_mse);
      if (!binary)
        (ko.Stream()) << "\n";

      KALDI_LOG<< "Write Mse layer to " << out_mse_filename << ".\n";
    } else {
      KALDI_WARN << "No MSE layer to be generated!";
    }

  } catch(const std::exception &e) {
    std::cout<< e.what() << std::endl;
    return -1;
  }
}

