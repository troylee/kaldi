/*
 * nnetbin/posnegbl-train.cc
 *
 * Based on nnetbin/nnet-train-xent-hardlab-perutt.cc
 *
 * Currently the input features and noises all must be MFCC_0_D_A.
 *
 * The <posnegbl> layer must be the first layer of the nnet.
 *
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-posnegbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform one iteration of Neural Network training by stochastic gradient descent.\n"
            "Usage:  posnegbl-train [options] <model-in> <feature-rspecifier> "
            "<noise-rspecifier> <alignments-rspecifier> [<model-out|output-wspecifier>]\n"
            "e.g.: \n"
            " posnegbl-train nnet.init scp:train.scp ark:noise.ark ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);

    bool cross_validate = false;
    po.Register("cross-validate", &cross_validate,
                "Perform cross-validation (don't backpropagate)");

    bool binary = false;
    po.Register(
        "binary",
        &binary,
        "Write the output model in binary format, only applicapble when the update-flag is model");

    std::string update_flag = "model";
    po.Register("update-flag", &update_flag,
                "Which parameter to update 'model' or 'noise'");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

    bool shared_var = true;
    po.Register(
        "shared-var",
        &shared_var,
        "Whether after compensation the variances are constrained to be the same");

    BaseFloat positive_var_weight = 1.0;
    po.Register(
        "positive-var-weight",
        &positive_var_weight,
        "Only effective when shared-var=true, the weight ratio for positive variance");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string learn_factors = "";
    po.Register("learn-factors", &learn_factors, "Learning rate factors (can control which layer to update)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad, "Average the gradient in the bunch");

    po.Read(argc, argv);

    if (!cross_validate && update_flag != "model" && update_flag != "noise") {
      KALDI_ERR<< "Unrecognized update flag: " << update_flag;
    }

    if (po.NumArgs() != (cross_validate ? 4 : 5)) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_in_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        noise_rspecifier = po.GetArg(3),
        alignments_rspecifier = po.GetArg(4), output_filename = po.GetOptArg(5);

    // currently no feature transform is supported
    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(nnet_in_filename);

    if (!cross_validate) {
      // only allow the first layer, which is <posnegbl> to be updated
      if(learn_factors!=""){
        nnet.SetLearnRate(learn_rate, learn_factors.c_str());
      }
      nnet.SetMomentum(momentum);
      nnet.SetL2Penalty(l2_penalty);
      nnet.SetL1Penalty(l1_penalty);
      nnet.SetAverageGrad(average_grad);
    }

    if (nnet.Layer(0)->GetType() != Component::kPosNegBL) {
      KALDI_ERR<< "The first layer is not <posnegbl> layer!";
    }
    PosNegBL *posnegbl_layer = static_cast<PosNegBL*> (nnet.Layer(0));
    if (!cross_validate) {
      posnegbl_layer->SetUpdateFlag(update_flag);
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    // only used when update_flag = "noise"
    DoubleVectorWriter noise_writer;
    if (!cross_validate && update_flag == "noise") {
      noise_writer.Open(output_filename);
    }

    Xent xent;

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, glob_err;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (cross_validate?"CV":("TRAINING "+update_flag)) << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
    for (; !feature_reader.Done(); /*feature_reader.Next()*/) {
      std::string key = feature_reader.Key();

      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
      } else {

        // load noise parameters
        if (!noiseparams_reader.HasKey(key + "_mu_h")
            || !noiseparams_reader.HasKey(key + "_mu_z")
            || !noiseparams_reader.HasKey(key + "_var_z")) {
          KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        }

        Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
        Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
        Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));

        posnegbl_layer->SetNoise(compensate_var, mu_h, mu_z, var_z, positive_var_weight);

        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        // std::cout << mat;

        if ((int32) alignment.size() != mat.NumRows()) {
          KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        if (num_done % 1000 == 0) {
          std::cout << num_done << ", " << std::flush;
        }
        num_done++;

        // push features to GPU
        feats.CopyFromMat(mat);

        nnet_transf.Feedforward(feats, &feats_transf);
        nnet.Propagate(feats_transf, &nnet_out);

        xent.EvalVec(nnet_out, alignment, &glob_err);

        if(!cross_validate) {
          nnet.Backpropagate(glob_err, NULL);
        }

        if(!cross_validate && update_flag=="noise") {
          posnegbl_layer->GetNoise(mu_h, mu_z, var_z);
          noise_writer.Write(key+"_mu_h", mu_h);
          noise_writer.Write(key+"_mu_z", mu_z);
          noise_writer.Write(key+"_var_z", var_z);
        }

        tot_t += mat.NumRows();
      }

      Timer t_features;
      feature_reader.Next();
      time_next += t_features.Elapsed();

    }

    if (!cross_validate && update_flag == "model") {
      nnet.Write(output_filename, binary);
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (cross_validate?"CV":("TRAINING "+update_flag)) << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
