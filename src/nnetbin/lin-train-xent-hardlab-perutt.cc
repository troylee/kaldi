// nnetbin/lin-train-xent-hardlab-perutt.cc
/*
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
            "Usage: lin-train-xent-hardlab-perutt [options] <model-in> <feature-rspecifier> <alignments-rspecifier>"
            "<weights-wspecifier> <bias-wspecifier>\n"
            "e.g.: \n"
            " lin-train-xent-hardlab-perutt --average-grad=true lin.init scp:train.scp "
            "ark:train.ali ark:weights.ark ark:bias.ark\n";

    ParseOptions po(usage);

    int32 max_iters = 10;
    po.Register("max-iters", &max_iters,
                "Maixmum number of training iterations");

    BaseFloat learn_rate_init_val = 0.001,
        learn_rate_end_val = 0.00001,
        start_halving_inc = 0.5,
        halving_factor = 0.5,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate-init-val", &learn_rate_init_val,
                "Initial learning rate");
    po.Register("learn-rate-end-val", &learn_rate_end_val,
                "Stopping criterion");
    po.Register("start-halving-inc", &start_halving_inc,
                "Learning rate halving starting condition");
    po.Register("halving-factor", &halving_factor,
                "Learning rate halving factor");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    bool average_grad = false;
    po.Register("average-grad", &average_grad, "Average the gradient");

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

    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);

    if (nnet.Layer(0)->GetType() != Component::kLinBL) {
      KALDI_ERR<< "The first layer is not <linbl> layer!";
    }
    LinBL *lin = static_cast<LinBL*>(nnet.Layer(0));

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    BaseFloatMatrixWriter weight_writer(weight_wspecifier);
    BaseFloatVectorWriter bias_writer(bias_wspecifier);

    Xent xent;

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, glob_err, prev_weight;
    CuVector<BaseFloat> prev_bias;

    Matrix<BaseFloat> lin_weight;
    Vector<BaseFloat> lin_bias;

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
    for (; !feature_reader.Done();) {
      std::string key = feature_reader.Key();

      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        Timer lin_timer;
        /*
         * LIN Training for the current utterance
         */
        KALDI_LOG << "============================================================";
        KALDI_LOG << "*** Utterance: " << key;
        KALDI_LOG << "Training started ...";

        /* initialize variables */
        BaseFloat acc, acc_new, acc_prev;
        BaseFloat learn_rate = learn_rate_init_val;

        /* Feature preparation */
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        // std::cout << mat;

        if ((int32) alignment.size() != mat.NumRows()) {
          KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        KALDI_LOG << "[" << mat.NumRows() << " frames]";

          /* forward feats through input transform if has */
        feats.CopyFromMat(mat);  // push features to GPU
        nnet_transf.Feedforward(feats, &feats_transf);

        /* initialize si nnet */
        lin->SetToIdentity();
        nnet.SetLearnRate(learn_rate, learn_factors.c_str());

        /* pre-run cross-validation */
        nnet.Propagate(feats_transf, &nnet_out);
        xent.Reset();
        xent.EvalVec(nnet_out, alignment, &glob_err);
        acc=xent.GetFrameAccuracy();
        KALDI_LOG << "CROSSVAL PRERUN ACCURACY " << acc ;

        for(int32 iter=0; iter < max_iters; ++iter){
          // keep a copy of the current model
          prev_weight.CopyFromMat(lin->GetLinearityWeight());
          prev_bias.CopyFromVec(lin->GetBiasWeight());

          // update the model
          nnet.Backpropagate(glob_err, NULL);

          // cross-validation
          nnet.Propagate(feats_transf, &nnet_out);
          xent.Reset();
          xent.EvalVec(nnet_out, alignment, &glob_err);
          acc_new=xent.GetFrameAccuracy();

          acc_prev = acc;
          if(acc_new > acc){
            // accept the weight
            acc = acc_new;

            KALDI_LOG << "ITERATION " << iter << ": LRATE " << learn_rate
                << " CROSSVAL ACCURACY " << acc_new << " [accepted]";
          }else{
            // reject and revert back the weight
            lin->SetLinearityWeight(prev_weight, false);
            lin->SetBiasWeight(prev_bias);

            KALDI_LOG << "ITERATION " << iter << ": LRATE " << learn_rate
                << " CROSSVAL ACCURACY " << acc_new << " [rejected]";
          }

          if (learn_rate < learn_rate_end_val){
            KALDI_LOG << "Too small learning rate " << learn_rate << "!";
            break;
          }

          if (acc < acc_prev + start_halving_inc){
            learn_rate = learn_rate * halving_factor;
            nnet.SetLearnRate(learn_rate, learn_factors.c_str());
          }
        }

        /* save the LIN */
        (lin->GetLinearityWeight()).CopyToMat(&lin_weight);
        (lin->GetBiasWeight()).CopyToVec(&lin_bias);

        weight_writer.Write(key, lin_weight);
        bias_writer.Write(key, lin_bias);

        KALDI_LOG << "Training finished in " << lin_timer.Elapsed() << "s.";
        KALDI_LOG << "============================================================";


        ++num_done;
        if(num_done % 1000 == 0){
          KALDI_LOG << "[" << num_done << " done]";
        }

      }

      feature_reader.Next();
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
