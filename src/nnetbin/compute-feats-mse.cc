// nnetbin/compute-feats-mse.cc

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
        "Compute the MSE between two sets of features.\n"
            "Usage:  compute-feats-mse [options] <feature-rspecifier> <targets-rspecifier>\n"
            "e.g.: \n"
            " compute-feats-mse scp:train.scp ark:targets.scp\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        targets_rspecifier = po.GetArg(2);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);

    Mse mse;

    CuMatrix<BaseFloat> feats, targets, glob_err;

    Timer tim;
    double time_next = 0;

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;

    while (!feature_reader.Done() && !targets_reader.Done()) {
      // get the keys
      std::string tgt_key = targets_reader.Key();
      std::string fea_key = feature_reader.Key();
      // skip feature matrix with no targets
      while (fea_key != tgt_key) {
        KALDI_WARN<< "No targets for: " << fea_key;
        num_no_tgt_mat++;
        if (!feature_reader.Done()) {
          feature_reader.Next();
          fea_key = feature_reader.Key();
        }
      }
        // now we should have a pair
      if (fea_key == tgt_key) {
        // get feature tgt_mat pair
        const Matrix<BaseFloat> &fea_mat = feature_reader.Value();
        const Matrix<BaseFloat> &tgt_mat = targets_reader.Value();
        // chech for dimension
        if (tgt_mat.NumRows() != fea_mat.NumRows()) {
          KALDI_WARN<< "Alignment has wrong size "<< (tgt_mat.NumRows()) << " vs. "<< (fea_mat.NumRows());
          num_other_error++;
          continue;
        }
          // push features/targets to GPU
        feats.CopyFromMat(fea_mat);
        targets.CopyFromMat(tgt_mat);
        mse.Eval(feats, targets, &glob_err);

        num_done++;
      }
      Timer t_features;
      feature_reader.Next();
      targets_reader.Next();
      time_next += t_features.Elapsed();
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< "COMPUTATION" << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_tgt_mat
    << " with no tgt_mats, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
