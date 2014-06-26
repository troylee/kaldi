// nnetbin/compute-mask-ratio.cc

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Check whether the given input masks are all one.\n"
            "Usage:  check-all-one-masks [options] <mask-rspecifier>\n"
            "e.g.: \n"
            " check-all-one-masks scp:train.scp\n";

    ParseOptions po(usage);

    bool print_per_file = false;
    po.Register("print-per-file", &print_per_file, "Whether to print per file ratio.");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    CuMatrix<BaseFloat> feats;

    Timer tim;
    double time_next = 0;

    int32 num_done = 0;
    double sum = 0.0, sum2 = 0.0, ratio = 0.0;

    while (!feature_reader.Done()) {
      // get the keys
      std::string fea_key = feature_reader.Key();
      // get feature tgt_mat pair
      const Matrix<BaseFloat> &fea_mat = feature_reader.Value();

      ratio = fea_mat.Sum() *1.0 / (fea_mat.NumRows() * fea_mat.NumCols());

      sum += ratio;

      sum2 += (ratio * ratio);

      num_done++;

      if (print_per_file) {
        KALDI_LOG << fea_key << " " << ratio;
      }

      feature_reader.Next();
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< "COMPUTATION" << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, average mask ratio: " << sum/num_done
    << ", standard deviation: " << sqrt((sum2 - sum*sum/num_done) / num_done);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
