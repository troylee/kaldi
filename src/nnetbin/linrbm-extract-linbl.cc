// nnetbin/linrbm-extract-linbl.cc

/*
 * Extract the LinBL layer from a given LinRbm model
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-linrbm.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Extract the LinBL layer from a LinRbm.\n"
            "Usage:  linrbm-extract-linbl [options] <linrbm-in> <linbl-out>\n"
            "e.g.:\n"
            " linrbm-extract-linbl --binary=false linrbm.mdl linbl.mdl\n";

    ParseOptions po(usage);

    bool binary_write = false;
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    Nnet nnet;
    nnet.Read(model_in_filename);

    if (nnet.Layer(0)->GetType() != Component::kLinRbm) {
      KALDI_ERR<< "The first layer is not <linrbm> layer!";
    }
    LinRbm *linrbm = static_cast<LinRbm*>(nnet.Layer(0));

    LinBL linbl(linrbm->InputDim(), linrbm->InputDim(), NULL);
    linbl.SetLinearityWeight(linrbm->GetLinLinearityWeight(), false);
    linbl.SetBiasWeight(linrbm->GetLinBiasWeight());
    linbl.SetLinBLType(linrbm->GetLinRbmType(), linrbm->GetLinRbmNumBlks(),
                       linrbm->GetLinRbmBlkDim());

    {
      Output ko(model_out_filename, binary_write);
      linbl.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

