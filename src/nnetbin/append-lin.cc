// nnetbin/append-lin.cc

/*
 * Append LIN network to the original nnet.
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-maskedbl.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Append a LIN to the existing nnet.\n"
            "Usage:  append-lin [options] <nnet-in> <nnet-out>\n"
            "e.g.:\n"
            " append-lin --binary=false nnet.mdl nnet_lin.mdl\n";

    ParseOptions po(usage);

    bool binary_write = false;
    po.Register("binary", &binary_write, "Write output in binary mode");

    bool diagblock = false;
    int32 blockdim = 123;
    int32 numblocks = 9;
    po.Register("diagonal-block", &diagblock, "Use diagonal block LIN");
    po.Register("block-dim", &blockdim,
                "Dimension of each block (square block)");
    po.Register("num-blocks", &numblocks, "Number of blocks in the LIN");

    bool shared = false;
    po.Register("shared", &shared, "Use shared diagonal block LIN");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    Nnet nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    int32 in_dim = nnet.InputDim();

    {
      Output ko(model_out_filename, binary_write);

      if (diagblock) {
        if (blockdim * numblocks != in_dim) {
          KALDI_ERR<< "Invalid block configuration: [dim: " << blockdim << ", num: " << numblocks << "], input dim: " << in_dim;
        }
        MaskedBL mbl(in_dim, in_dim, NULL);
        mbl.SetToIdentity();
        Matrix<BaseFloat> mask(in_dim, in_dim, kSetZero);
        for(int32 i=0, offset=0; i<numblocks; ++i, offset+=blockdim) {
          for(int32 j=0; j<blockdim; ++j) {
            for(int32 k=0; k<blockdim; ++k) {
              mask(offset+j, offset+k)=1.0;
            }
          }
        }
        mbl.SetMask(mask);

        if (shared){
          mbl.SetSharing(numblocks, blockdim, blockdim);
        }

        mbl.Write(ko.Stream(), binary_write);

      } else {
        BiasedLinearity bl(in_dim, in_dim, NULL);
        bl.SetToIdentity();

        bl.Write(ko.Stream(), binary_write);
      }

      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

