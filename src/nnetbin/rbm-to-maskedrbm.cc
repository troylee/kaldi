// gmmbin/rbm-to-maskedrbm.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-rbm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert RBM to MaskedRbm\n"
            "Usage:  rbm-to-maskedrbm [options] <rbm-in> <nnet-out>\n"
            "e.g.:\n"
            " rbm-to-maskedrbm --binary=false rbm.mdl makedrbm.mdl\n";

    bool binary_write = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    int32 filter_height = 8;
    int32 filter_width = 8;
    int32 filter_step = 2;

    int32 input_height = 9;  // equals to the number of frames
    int32 input_width = 39;  // equals to the feature dim

    po.Register("filter-height", &filter_height, "Height of the filter");
    po.Register("filter-width", &filter_width, "Width of the filter");
    po.Register("filter-step", &filter_step,
                "Distance between two adjacent filters");
    po.Register("input-height", &input_height, "Height of the input");
    po.Register("input-width", &input_width, "Width of the input");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);


    MaskedRbm rbm(0, 0, NULL);
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      rbm.ReadRbm(ki.Stream(), binary_read);
    }

    int32 outdim = rbm.OutputDim(), indim = rbm.InputDim();
    Matrix<BaseFloat> mask(outdim, indim, kSetZero);

    KALDI_ASSERT(input_height * input_width == indim);

    int32 num_row_filters = input_height / filter_step;
    int32 num_col_filters = input_width / filter_step;
    if (num_row_filters * num_col_filters != outdim) {
      KALDI_ERR<< "The number of RBM hidden units should equal to the number of filters: " << outdim << " v.s. " << num_row_filters * num_col_filters;
    }

    int32 s_w = -filter_width / 2, s_h = -filter_height / 2;
    int32 e_w = s_w + filter_width, e_h = s_h + filter_height;

    for (int32 i = 0, k = 0; i < input_height; i += filter_step) {
      for (int32 j = 0; j < input_width; j += filter_step, ++k) {
        for (int32 r = s_h; r < e_h; ++r) {
          for (int32 c = s_w; c < e_w; ++c) {
            if (i + r >= 0 && i + r < input_height && j + c >= 0
                && j + c < input_width) {
              mask(k, (i + r) * input_width + j + c) = 1.0;
            }
          }
        }
      }
    }

    rbm.SetMask(mask);

    {
      Output ko(model_out_filename, binary_write);
      rbm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

