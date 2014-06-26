// nnetbin/scale-nnet.cc

/*
 * Scale the weight of specific layers.
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-biasedlinearity.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Extract layers from an existing nnet.\n"
            "Usage:  scale-nnet [options] <nnet-in> <nnet-out> <layer-ids ...>\n"
            "e.g.:\n"
            " sub-nnet --binary=false --scale=0.5 nnet.mdl nnet_sub.mdl 1\n";

    bool binary_write = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    BaseFloat scale = 1.0;
    po.Register("scale", &scale, "Scaling coefficient");

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

    /* scaling the specific layers */
    for (int32 i = 3; i <= num_args; ++i) {
      std::string layer = po.GetArg(i);
      KALDI_LOG<< "Write out layer " << layer;
      int lid = atoi(layer.c_str());
      if((nnet.Layer(lid))->GetType()!=Component::kBiasedLinearity ){
        KALDI_WARN << "Layer " << lid << " is not <biasedlinearity>, ignored.";
      }
      BiasedLinearity *bl = static_cast<BiasedLinearity*>(nnet.Layer(lid));
      CuMatrix<BaseFloat> weight;
      CuVector<BaseFloat> bias;
      weight.CopyFromMat(bl->GetLinearityWeight());
      bias.CopyFromVec(bl->GetBiasWeight());

      weight.Scale(scale);
      bias.Scale(scale);

      bl->SetLinearityWeight(weight, false);
      bl->SetBiasWeight(bias);
    }

    nnet.Write(model_out_filename, binary_write);

    KALDI_LOG<< "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

