// nnetbin/lin-train-perutt-single-iter.cc
/*
 * Single iteration of training.
 * Train a LIN per utterance and save the transforms in archive file.
 * Due to the only utterance, train and cv are the same.
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Train LIN for each utterance. It is better to enable average-grad for utterance-based training.\n"
            "Usage: lin-train-perutt-single-iter [options] <model-in> <feature-rspecifier> <alignments-rspecifier>"
            "<weights-wspecifier> <bias-wspecifier>\n"
            "e.g.: \n"
            " lin-train-perutt-single-iter --average-grad=true lin.init scp:train.scp ark:train.ali "
            "ark:weights.ark ark:bias.ark\n";

    ParseOptions po(usage);

    BaseFloat learn_rate = 0.001,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate,
                "Learning rate");

    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    std::string weight_init = "", bias_init = "";
    po.Register("weight-init", &weight_init, "Initial weight archive");
    po.Register("bias-init", &bias_init, "Initial bias archive");

    bool average_grad = false;
    po.Register("average-grad", &average_grad, "Average the gradent or not");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string si_model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        weight_wspecifier = po.GetArg(4),
        bias_wspecifier = po.GetArg(5);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(si_model_filename);

    // construct the learn factors, only learn the first LIN layer,
    // i.e. "1,0,0,0,...,0"
    std::string learn_factors = "1";
    for (int32 i = 1; i < nnet.LayerCount(); ++i) {
      if (nnet.Layer(i)->IsUpdatable())
        learn_factors += ",0";
    }

    nnet.SetAverageGrad(average_grad);
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);
    nnet.SetLearnRate(learn_rate, learn_factors.c_str());

    if (nnet.Layer(0)->GetType() != Component::kLinBL) {
      KALDI_ERR<< "The first layer is not <linbl> layer!";
    }
    LinBL *lin = static_cast<LinBL*>(nnet.Layer(0));

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    RandomAccessBaseFloatMatrixReader weight_reader(weight_init);
    RandomAccessBaseFloatVectorReader bias_reader(bias_init);

    BaseFloatMatrixWriter weight_writer(weight_wspecifier);
    BaseFloatVectorWriter bias_writer(bias_wspecifier);

    Xent xent;

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, glob_err;

    Matrix<BaseFloat> lin_weight;
    Vector<BaseFloat> lin_bias;

    Timer lin_timer;

    KALDI_LOG<< "============================================================";
    KALDI_LOG<< "Training started ...";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
    /* initialize variables */
    BaseFloat acc;

    for (; !feature_reader.Done();) {
      std::string key = feature_reader.Key();

      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
        continue;
      }

      /*
       * LIN Training for the current utterance
       */

      /* Feature preparation */
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const std::vector<int32> &alignment = alignments_reader.Value(key);

      // std::cout << mat;

      if ((int32) alignment.size() != mat.NumRows()) {
        KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
        num_other_error++;
        continue;
      }

        /* forward feats through input transform if has */
      feats.CopyFromMat(mat);  // push features to GPU
      nnet_transf.Feedforward(feats, &feats_transf);

      /* initialize si nnet */
      lin->SetToIdentity();
      if (weight_init != "" && weight_reader.HasKey(key)) {
        lin->SetLinearityWeight(weight_reader.Value(key), false);
      }
      if (bias_init != "" && bias_reader.HasKey(key)) {
        lin->SetBiasWeight(bias_reader.Value(key));
      }

      /* pre-run cross-validation */
      nnet.Propagate(feats_transf, &nnet_out);
      xent.Reset();
      xent.EvalVec(nnet_out, alignment, &glob_err);

      // update the model
      nnet.Backpropagate(glob_err, NULL);

      /* save the LIN */
      (lin->GetLinearityWeight()).CopyToMat(&lin_weight);
      (lin->GetBiasWeight()).CopyToVec(&lin_bias);

      weight_writer.Write(key, lin_weight);
      bias_writer.Write(key, lin_bias);

      /* test the updated model */
      nnet.Propagate(feats_transf, &nnet_out);
      xent.Reset();
      xent.EvalVec(nnet_out, alignment, &glob_err);
      acc = xent.GetFrameAccuracy();

      KALDI_LOG<< "*** Utterance: " << key << ", [" << mat.NumRows() << " frames][CV " << acc << "]";

      ++num_done;
      if (num_done % 1000 == 0) {
        KALDI_LOG<< "[" << num_done << " done]";
      }

      feature_reader.Next();
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";
    KALDI_LOG<< "Training finished in " << lin_timer.Elapsed() << "s.";
    KALDI_LOG<< "============================================================";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
