/*
 * vtsbin/kl-divergence-gauss.cc
 *
 *  Created on: Nov 12, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Compute the per Gaussian KL divergence between two AMs, which must have
 *  exactly the same structure.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"

#include "vts/vts-first-order.h"

int main(int argc, char* argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Compute the per Gaussian KL divergence between two AMs,"
            "The output stats are ordered by Gaussian IDs in the AM. \n"
            "Usage: kl-divergence-gauss [options] true-am-in estimate-am-in kl-stats-out\n";
    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string true_am_filename = po.GetArg(1), estimate_am_filename = po
        .GetArg(2), kl_stats_wxfilename = po.GetArg(3);

    AmDiagGmm true_am_gmm;
    {
      bool binary;
      Input ki(true_am_filename, &binary);
      true_am_gmm.Read(ki.Stream(), binary);
    }
    AmDiagGmm estimate_am_gmm;
    {
      bool binary;
      Input ki(estimate_am_filename, &binary);
      estimate_am_gmm.Read(ki.Stream(), binary);
    }

    // compute the mean and variance for each hidden unit
    // and compute the KL divergence
    Vector<double> klstats(true_am_gmm.NumGauss(), kSetZero);
    KALDI_ASSERT(true_am_gmm.NumPdfs() == estimate_am_gmm.NumPdfs());
    for (int32 c = 0, gid = 0; c < true_am_gmm.NumPdfs(); ++c) {

      DiagGmm *true_gmm = &(true_am_gmm.GetPdf(c));
      DiagGmmNormal true_ngmm(*true_gmm);

      DiagGmm *est_gmm = &(estimate_am_gmm.GetPdf(c));
      DiagGmmNormal est_ngmm(*est_gmm);

      KALDI_ASSERT(true_gmm->NumGauss() == est_gmm->NumGauss());
      for (int32 g = 0; g < true_gmm->NumGauss(); ++g, ++gid) {
        klstats(gid) = KLDivergenceDiagGaussian(Vector<double>(true_ngmm.means_.Row(g)),
                                                Vector<double>(true_ngmm.vars_.Row(g)),
                                                Vector<double>(est_ngmm.means_.Row(g)),
                                                Vector<double>(est_ngmm.vars_.Row(g)));
      }
    }

    {
      kaldi::Output ko(kl_stats_wxfilename, binary);
      klstats.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Written KL stats.";
    KALDI_LOG<< "Average KL Divergence among all Gaussians is: " << klstats.Sum()/ klstats.Dim();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

