// nnetbin/ubm-avg-likes.cc

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
        "Compute the average frame likelihoods of the UBM.\n"
        "Usage:  ubm-avg-likes [options] <model-in> <in-rspecifier>\n"
        "e.g.:\n"
        " ubm-avg-likes 1.mdl scp:feat.scp\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2);


    DiagGmm gmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      gmm.Read(ki.Stream(), binary_read);
    }

    std::cout << "Totally " << gmm.NumGauss() << " Gaussians.\n";
    std::cout << "Feature Dimension: " << gmm.Dim() << ".\n";

    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);

    int32 tot_utt = 0, tot_frames = 0;
    BaseFloat tot_likes = 0.0;

    for(;!feats_reader.Done(); feats_reader.Next()) {
      const Matrix<BaseFloat> &feats=feats_reader.Value();
      for(int32 r = 0 ; r < feats.NumRows(); ++r){
        tot_likes += gmm.LogLikelihood(feats.Row(r));
      }
      tot_frames += feats.NumRows();
      tot_utt += 1;
    }

    std::cout << "Computed " << tot_utt << " utterances, " << tot_frames
        << " frames, the average per-frame log-likelihood is " << tot_likes / tot_frames << "\n.";

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


