// nnetbin/gaussbl-create.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-gaussbl.h"

namespace kaldi {

// read in the biasedlinearity layer
bool ReadBiasedLinearityLayer(std::istream &is, bool binary,
                              Matrix<BaseFloat> &linearity,
                              Vector<BaseFloat> &bias) {

  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF)
    return false;

  ReadToken(is, binary, &token);
  Component::ComponentType comp_type = Component::MarkerToType(token);

  if (comp_type != Component::kBiasedLinearity) {
    KALDI_ERR<< "Layer type error! '<biasedlinearity>' is expected, but got '"
    << token << "'";
  }

  ReadBasicType(is, binary, &dim_out);
  ReadBasicType(is, binary, &dim_in);

  if (linearity.NumRows() != dim_out || linearity.NumCols() != dim_in) {
    linearity.Resize(dim_out, dim_in);
  }
  if (bias.Dim() != dim_out) {
    bias.Resize(dim_out);
  }

  linearity.Read(is, binary);
  bias.Read(is, binary);

  return true;

}

/*
 * weight and bias are initialised as the DBN input layer weights,
 * i.e. in the normalized feature space.
 *
 * w_dis = w ./ norm_std;
 * b_dis = b - w .* norm_mean ./ norm_std;
 *
 */
void ConvertWeightToOriginalSpace(int32 num_frames,
                                  const Vector<double> &norm_mean,
                                  const Vector<double> &norm_std,
                                  Matrix<BaseFloat> &weight,
                                  Vector<BaseFloat> &bias) {
  int32 feat_dim = norm_mean.Dim();
  double tmp = 0.0;
  KALDI_ASSERT(num_frames * feat_dim == weight.NumCols());

  for (int32 r = 0; r < weight.NumRows(); ++r) {
    tmp = 0.0;
    for (int32 c = 0; c < weight.NumCols(); ++c) {
      tmp += weight(r, c) * norm_mean(c % feat_dim) / norm_std(c % feat_dim);
      weight(r, c) = weight(r, c) / norm_std(c % feat_dim);
    }
    bias(r) = bias(r) - tmp;
  }

}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Create GaussBL layer from positive and negative GMMs and conventional NN layer weights\n"
            "Usage:  posnegbl-create [options] <pos-model-in> <neg-model-in>"
            " <ori-bl-layer> <model-out>\n"
            "e.g.:\n"
            " gaussbl-create --binary=false pos.mdl neg.mdl bl_layer nnet.mdl\n";

    bool binary_write = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    int32 num_frame = 9;
    po.Register("num-frame", &num_frame,
                "Number of frames for the input feature");

    int32 delta_order = 2;
    po.Register("delta-order", &delta_order,
                "Delta order of the input feature, 0-static, 1-delta, 2-acc");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");

    BaseFloat ceplifter = 22;
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    std::string back_nnet = "";
    po.Register(
        "back-nnet", &back_nnet,
        "Back-end nnet model file, to be concatenated after this front end");

    std::string cmvn_stats_rspecifier = "";
    po.Register("cmvn-stats", &cmvn_stats_rspecifier,
                "rspecifier for global CMVN feature normalization");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_model_filename = po.GetArg(1), neg_model_filename = po
        .GetArg(2), bl_layer_filename = po.GetArg(3), model_out_filename = po
        .GetArg(4);

    // positive AM Gmm
    AmDiagGmm pos_am_gmm;
    {
      bool binary;
      Input ki(pos_model_filename, &binary);
      pos_am_gmm.Read(ki.Stream(), binary);
    }

    // negative AM Gmm
    AmDiagGmm neg_am_gmm;
    {
      bool binary;
      Input ki(neg_model_filename, &binary);
      neg_am_gmm.Read(ki.Stream(), binary);
    }

    KALDI_ASSERT(pos_am_gmm.NumPdfs() == neg_am_gmm.NumPdfs());
    int32 num_pdfs = pos_am_gmm.NumPdfs();

    // biased linearity layer weight and bias
    Matrix<BaseFloat> linearity;
    Vector<BaseFloat> bias;
    {
      bool binary;
      Input ki(bl_layer_filename, &binary);
      if (!ReadBiasedLinearityLayer(ki.Stream(), binary, linearity, bias)) {
        KALDI_ERR<< "Load biased linearity layer from " << bl_layer_filename
        << " failed!";
      }
    }

        // converting the normalized weights to original feature space
        // if necessary
    Vector<double> norm_mean, norm_std;
    if (cmvn_stats_rspecifier != "") {
      // convert the models back to the original feature space
      RandomAccessDoubleMatrixReader cmvn_reader(cmvn_stats_rspecifier);
      if (!cmvn_reader.HasKey("global")) {
        KALDI_ERR<< "No normalization statistics available for key global";
      }
      const Matrix<double> &stats = cmvn_reader.Value("global");
      // convert stats to mean and std
      int32 dim = stats.NumCols() - 1;
      norm_mean.Resize(dim, kSetZero);
      norm_std.Resize(dim, kSetZero);
      double count = stats(0, dim);
      if (count < 1.0)
        KALDI_ERR<< "Insufficient stats for cepstral mean and variance normalization: "
        << "count = " << count;

      for (int32 i = 0; i < dim; ++i) {
        norm_mean(i) = stats(0, i) / count;
        norm_std(i) = sqrt((stats(1, i) / count) - norm_mean(i) * norm_mean(i));
      }

      // converting weights
      ConvertWeightToOriginalSpace(num_frame, norm_mean, norm_std, linearity,
                                   bias);
    }

    Nnet back;
    if (back_nnet != "") {
      back.Read(back_nnet);
    }

    int32 indim = pos_am_gmm.Dim();
    int32 outdim = num_pdfs;

    GaussBL *layer = new GaussBL(indim, outdim, NULL);
    layer->CreateModel(num_frame, delta_order, num_cepstral, num_fbank,
                       ceplifter,
                       pos_am_gmm,
                       neg_am_gmm,
                       linearity, bias);

    {
      Output ko(model_out_filename, binary_write);
      layer->Write(ko.Stream(), binary_write);
      if (back_nnet != "") {
        back.Write(ko.Stream(), binary_write);
      }
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

