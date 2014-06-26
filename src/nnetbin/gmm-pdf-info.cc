// nnetbin/gmm-pdf-info.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Write to standard output number of Gaussians in a specific PDF\n"
        "Usage:  gmm-pdf-info [options] <model-in> [<gmm-out>]\n"
        "e.g.:\n"
        " gmm-pdf-info --pdf=0 1.mdl\n";
    
    ParseOptions po(usage);
    
    int32 pdf=0;
    po.Register("pdf", &pdf, "PDF id to inquiry");

    po.Read(argc, argv);

    if (po.NumArgs() != 1 && po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        out_filename = po.GetOptArg(2);


    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    std::cout << "In PDF " << pdf << ", there are " << am_gmm.NumGaussInPdf(pdf) << " Gaussians.\n";

    if (out_filename != ""){
      Output ko(out_filename, false);

      const DiagGmm &gmm = am_gmm.GetPdf(pdf);

      int32 num_comp = gmm.NumGauss(),
          feat_dim = gmm.Dim();

      const Vector<BaseFloat> &weights = gmm.weights();
      weights.Write(ko.Stream(), false);

      Matrix<BaseFloat> mat(num_comp, feat_dim);
      gmm.GetMeans(&mat);
      mat.Write(ko.Stream(), false);

      gmm.GetVars(&mat);
      mat.Write(ko.Stream(), false);

      std::cout << "Write GMM into " << out_filename << ".\n";
    }

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


