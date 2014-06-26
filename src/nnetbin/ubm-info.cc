// nnetbin/ubm-info.cc

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
        "Write to standard output number of Gaussians in an UBM\n"
        "Usage:  ubm-info [options] <model-in> [<gmm-out>]\n"
        "e.g.:\n"
        " ubm-info 1.mdl\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 1 && po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        out_filename = po.GetOptArg(2);


    DiagGmm gmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      gmm.Read(ki.Stream(), binary_read);
    }

    std::cout << "Totally " << gmm.NumGauss() << " Gaussians.\n";
    std::cout << "Feature Dimension: " << gmm.Dim() << ".\n";

    if (out_filename != ""){
      Output ko(out_filename, false);

      int32 num_comp = gmm.NumGauss(),
          feat_dim = gmm.Dim();

      const Vector<BaseFloat> &weights = gmm.weights();
      weights.Write(ko.Stream(), false);

      Matrix<BaseFloat> mat(num_comp, feat_dim);
      gmm.GetMeans(&mat);
      mat.Write(ko.Stream(), false);

      gmm.GetVars(&mat);
      mat.Write(ko.Stream(), false);

      std::cout << "Write UBM into " << out_filename << ".\n";
    }

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


