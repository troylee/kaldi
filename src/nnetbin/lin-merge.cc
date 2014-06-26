// nnetbin/lin-merge.cc

/*
 * Merge all the LIN xforms to a weight archive and a bias arvhive.
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Merge LIN xforms to a weight archive and a bias archive.\n"
            "Usage:  lin-merge [options] <xform-rspecifier> <weight-wspecifier> <bias-wspecifier>\n"
            "e.g.:\n"
            " lin-merge ark:lin_xform.ark ark:lin_weight.ark ark:lin_bias.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string lin_list_rspecifier = po.GetArg(1),
        weight_out_wspecifier = po.GetArg(2),
        bias_out_wspecifier = po.GetArg(3);

    SequentialTokenReader lin_reader(lin_list_rspecifier);
    BaseFloatMatrixWriter weight_writer(weight_out_wspecifier);
    BaseFloatVectorWriter bias_writer(bias_out_wspecifier);

    Matrix<BaseFloat> lin_weight;
    Vector<BaseFloat> lin_bias;

    int32 num_tot = 0, num_done = 0, num_err = 0;
    for (; !lin_reader.Done(); lin_reader.Next(), ++num_tot) {
      std::string key = lin_reader.Key();
      std::string file = lin_reader.Value();

      Nnet nnet;
      {
        bool binary_read;
        Input ki(file, &binary_read);
        nnet.Read(ki.Stream(), binary_read);
      }
      if (nnet.Layer(0)->GetType() != Component::kLinBL) {
        KALDI_WARN<< key << " is not <linbl> layer, ignored.";
        ++num_err;
        continue;
      }
      LinBL *lin = static_cast<LinBL*>(nnet.Layer(0));
      (lin->GetLinearityWeight()).CopyToMat(&lin_weight);
      (lin->GetBiasWeight()).CopyToVec(&lin_bias);

      weight_writer.Write(key, lin_weight);
      bias_writer.Write(key, lin_bias);
      ++num_done;

    }

    KALDI_LOG<< "Totally " << num_tot << " items in the list file, "
        << num_done << " successfully done, "
        << num_err << " failed.";
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

