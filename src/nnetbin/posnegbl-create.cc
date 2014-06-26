// nnetbin/posnegbl-create.cc

// Copyright 2012  Karel Vesely

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
#include "nnet/nnet-posnegbl.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Create PosNegBL layer from postive and negative GMMs together with other statistics\n"
            "Usage:  posnegbl-create [options] <pos-model-in> <neg-model-in>"
            " <pos2neg-prior-in> <var-scale-in> <model-out>\n"
            "e.g.:\n"
            " posnegbl-create --binary=false pos.mdl neg.mdl pos2neg.stats var_scale.stats nnet.mdl\n";

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

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_model_filename = po.GetArg(1), neg_model_filename = po
        .GetArg(2), pos2neg_prior_filename = po.GetArg(3), var_scale_filename =
        po.GetArg(4), model_out_filename = po.GetArg(5);

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

    // positive to negative prior ratio
    Vector<double> pos2neg_log_prior_ratio(num_pdfs, kSetZero);
    Matrix<double> prior_stats;
    {
      bool binary;
      Input ki(pos2neg_prior_filename, &binary);
      prior_stats.Read(ki.Stream(), binary);
      KALDI_ASSERT(
                   prior_stats.NumRows()==2 && prior_stats.NumCols()==num_pdfs);
    }
    for (int32 i = 0; i < num_pdfs; ++i) {
      pos2neg_log_prior_ratio(i) = log(prior_stats(0, i) / prior_stats(1, i));
    }

    // variance scale factors
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == num_pdfs);
    }

    Nnet back;
    if (back_nnet != "") {
      back.Read(back_nnet);
    }

    int32 indim = pos_am_gmm.Dim();
    int32 outdim = num_pdfs;

    PosNegBL *layer = new PosNegBL(indim, outdim, NULL);
    layer->CreateModel(num_frame, delta_order, num_cepstral, num_fbank,
                       ceplifter,
                       pos2neg_log_prior_ratio, var_scale,
                       pos_am_gmm,
                       neg_am_gmm);

    {
      Output ko(model_out_filename, binary_write);
      layer->Write(ko.Stream(), binary_write);
      if(back_nnet!=""){
        back.Write(ko.Stream(), binary_write);
      }
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

