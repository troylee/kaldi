// nnetbin/sub-nnet.cc

/*
 * Extract the specified layers from a nnet .
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Extract layers from an existing nnet.\n"
        "Usage:  sub-nnet [options] <nnet-in> <nnet-out> <layer-ids ...>\n"
        "e.g.:\n"
        " sub-nnet --binary=false nnet.mdl nnet_sub.mdl\n";


    bool binary_write = false;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);

    int32 num_args = po.NumArgs();

    if (num_args < 3) {
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

    {
      Output ko(model_out_filename, binary_write);

      for (int32 i=3; i<=num_args; ++i){
        std::string layer = po.GetArg(i);
        KALDI_LOG << "Write out layer " << layer;
        (nnet.Layer(atoi(layer.c_str())))->Write(ko.Stream(), binary_write);
      }
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


