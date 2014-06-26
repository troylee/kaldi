// nnetbin/linrbm-init.cc

/*
 * Initialize a LIN NN with SI NN.
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-linrbm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Append a LIN to the existing RBM.\n"
            "Usage:  linrbm-init [options] <rbm-in> <rbm-out>\n"
            "e.g.:\n"
            " linrbm-init --binary=false rbm.mdl rbm_lin.mdl\n";

    ParseOptions po(usage);

    bool binary_write = false;
    po.Register("binary", &binary_write, "Write output in binary mode");

    int32 lin_type = 0, num_blks = 0, blk_dim = 0;
    po.Register(
        "lin-type",
        &lin_type,
        "LIN type: [0 - standard BL; 1 - diagonal BL; 2 - block diagonal BL; 3 - shared block diagonal BL]");
    po.Register("num-blks", &num_blks, "Number of blocks for type 2 and 3");
    po.Register("blk-dim", &blk_dim, "Block dimension for type 2 and 3");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    LinRbm linrbm(0,0,NULL);
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      linrbm.ReadRbm(ki.Stream(), binary_read);
    }

    linrbm.SetToIdentity();
    if(linrbm.GetLinRbmType() != lin_type){
      linrbm.SetLinRbmType(lin_type, num_blks, blk_dim);
    }

    {
      Output ko(model_out_filename, binary_write);
      linrbm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

