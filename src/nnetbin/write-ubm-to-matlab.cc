// nnetbin/write-ubm-to-matlab.cc
/*
 * Created on: Sept. 5, 2013
 *     Author: Troy Lee (troy.lee2008@gmail.com)
 *
 */

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Write the diagonal GMM UBM model to Matlab file.\n"
            "Usage: write-ubm-to-matlab [options] <model-file> <out-file>\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        out_filename = po.GetArg(2);

    DiagGmm ubm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      ubm.Read(ki.Stream(), binary_read);
    }
    {
      Output ko(out_filename, false);

      WriteToken(ko.Stream(), false, "gmm_wt=");
      (ubm.weights()).Write(ko.Stream(), false);
      WriteToken(ko.Stream(), false, ";");

      Matrix<BaseFloat> mat(ubm.NumGauss(), ubm.Dim(), kSetZero);
      WriteToken(ko.Stream(), false, "gmm_means=reshape(");
      ubm.GetMeans(&mat);
      mat.Write(ko.Stream(), false);
      WriteToken(ko.Stream(), false, ",");
      WriteBasicType(ko.Stream(), false, ubm.Dim());
      WriteToken(ko.Stream(), false, ",");
      WriteBasicType(ko.Stream(), false, ubm.NumGauss());
      WriteToken(ko.Stream(), false, ");");

      WriteToken(ko.Stream(), false, "gmm_vars=reshape(");
      ubm.GetVars(&mat);
      mat.Write(ko.Stream(), false);
      WriteToken(ko.Stream(), false, ",");
      WriteBasicType(ko.Stream(), false, ubm.Dim());
      WriteToken(ko.Stream(), false, ",");
      WriteBasicType(ko.Stream(), false, ubm.NumGauss());
      WriteToken(ko.Stream(), false, ");");

    }

    KALDI_LOG<< "Written UBM to " << out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

