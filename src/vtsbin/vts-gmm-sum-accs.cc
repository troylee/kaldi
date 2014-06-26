/*
 * vtsbin/vts-gmm-sum-accs.cc
 *
 *  Created on: Nov 25, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */


#include "util/common-utils.h"
#include "vts/vts-accum-am-diag-gmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for VAT training.\n"
        "Usage: vts-gmm-sum-accs [options] stats-out stats-in1 stats-in2 ...\n";

    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_out_filename = po.GetArg(1);
    kaldi::Vector<double> transition_accs;
    kaldi::VtsAccumAmDiagGmm gmm_accs;

    int num_accs = po.NumArgs() - 1;
    for (int i = 2, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read, true /*add read values*/);
      gmm_accs.Read(ki.Stream(), binary_read, true /*add read values*/);
    }

    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Summed " << num_accs << " stats, total count "
              << gmm_accs.TotCount() << ", avg like/frame "
              << (gmm_accs.TotLogLike() / gmm_accs.TotCount());
    KALDI_LOG << "Total count of stats is " << gmm_accs.TotStatsCount();
    KALDI_LOG << "Written stats to " << stats_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

