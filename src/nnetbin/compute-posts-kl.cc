// nnetbin/compute-posts-kl.cc

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Compute the KL divergence ( sum(P*ln(P/Q)) ) between two sets of posteriors per frame.\n"
            "Usage:  compute-posts-kl [options] <P-rspecifier> <Q-rspecifier> <KL-wspecifier>\n"
            "e.g.: \n"
            " compute-posts-kl scp:p_post.scp scp:q_post.scp ark:kl.ark\n";

    ParseOptions po(usage);

    BaseFloat zero_value = 1e-6; // values below this one are treated as 0
    po.Register("zero-value", &zero_value, "Zero value used for float numbers.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string p_rspecifier = po.GetArg(1),
        q_rspecifier = po.GetArg(2),
        kl_wspecifier = po.GetArg(3);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    kaldi::int64 tot_frames = 0;

    SequentialBaseFloatMatrixReader p_reader(p_rspecifier);
    RandomAccessBaseFloatMatrixReader q_reader(q_rspecifier);

    BaseFloatVectorWriter kl_writer(kl_wspecifier);

    double sum = 0.0, sum2 = 0.0, sum_utt_avg = 0.0;

    Timer tim;

    int32 num_done = 0, num_no_q = 0, num_other_error = 0;

    for (; !p_reader.Done(); p_reader.Next()) {
      // get the keys
      std::string key = p_reader.Key();

      if (!q_reader.HasKey(key)) {
        KALDI_WARN<< "Utterance " << key << " doesn't have corresponding Q proabilities.";
        ++num_no_q;
        continue;
      }

      // get feature tgt_mat pair
      const Matrix<BaseFloat> &p_post = p_reader.Value();
      const Matrix<BaseFloat> &q_post = q_reader.Value(key);
      // chech for dimension
      if (p_post.NumRows() != q_post.NumRows() || p_post.NumCols() != q_post.NumCols()) {
        KALDI_WARN<< "Posteriors have wrong sizes ["<< p_post.NumRows() << "," << p_post.NumCols()
            << "] vs. ["<< q_post.NumRows() << "," << q_post.NumCols() <<"].";
        num_other_error++;
        continue;
      }

      Vector<BaseFloat> stats(p_post.NumRows(), kSetZero);
      double utt_sum = 0.0, utt_sum2 = 0.0;
      for (int32 r = 0; r < p_post.NumRows(); ++r){
        stats(r)=0.0;
        for(int32 c =0 ; c< p_post.NumCols(); ++c){
          if (p_post(r, c) >= zero_value && q_post(r, c) >= zero_value){
            stats(r) += p_post(r,c) * log(p_post(r,c)/q_post(r,c));
          }
        }

        utt_sum += stats(r);
        utt_sum2 += stats(r) * stats(r);
        ++tot_frames;
      }

      sum += utt_sum;
      sum2 += utt_sum2;
      sum_utt_avg += (utt_sum/p_post.NumRows());

      kl_writer.Write(key, stats);

      num_done++;

    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< "COMPUTATION" << " FINISHED "
    << tim.Elapsed() << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_q
    << " with no Q posteriors, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< "Per-Frame KL Divergence Mean is: " << sum/tot_frames
        << ", Standard deviation is: " << sqrt((sum2 - sum/tot_frames)/tot_frames);
    KALDI_LOG << "Per-Utterance Average KL Divergence is: " << sum_utt_avg / num_done;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
