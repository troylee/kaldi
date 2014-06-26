/*
 * nnetbin/train-noise-cmvn.cc
 *
 * Based on nnetbin/nnet-train-xent-hardlab-perutt.cc
 *
 * Currently the input features and noises all must be FBanks.
 * CMVN must be either "global" key ark file or a single stats file.
 *
 * The input NNet is the common network augmented with an extra layer
 * "<cmvnbl> win_len feat_dim"
 *
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-cmvnbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
            "Usage:  train-noise-cmvn [options] <cmvn-stats> <model-in> <feature-rspecifier> "
            "<noise-rspecifier> <alignments-rspecifier> <out-wspecifier>\n"
            "e.g.: \n"
            " train-noise-cmvn cmvn_stats.ark nnet.init scp:train.scp ark:noise.ark ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);
    std::string update_flag = "cmvn";
    po.Register("update-flag", &update_flag,
                "Which parameter to update 'cmvn' or 'noise'");

    bool have_energy = true;
    po.Register(
        "have-energy", &have_energy,
        "Whether the feature has energy term, it will not be compensated");

    int32 num_fbank = 40;
    po.Register("num_fbank", &num_fbank, "Number of FBanks");

    int32 delta_order = 3;
    po.Register("delta-order", &delta_order,
                "Delta order of features, [1,2,3]");

    bool norm_vars = true;
    po.Register("norm-vars", &norm_vars, "If true, normalize variances");

    bool update_vars = true;
    po.Register("update-vars", &update_vars, "If true, estimate variances");

    bool cross_validate = false;
    po.Register("cross-validate", &cross_validate,
                "Perform cross-validation (don't backpropagate)");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    po.Read(argc, argv);

    if (!cross_validate && update_flag != "cmvn" && update_flag != "noise") {
      KALDI_ERR<< "Unrecognized update flag: " << update_flag;
    }

    if (po.NumArgs() != 6 - (cross_validate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string cmvn_rspecifier_or_rxfilename = po.GetArg(1),
        nnet_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        noise_rspecifier = po.GetArg(4),
        alignments_rspecifier = po.GetArg(5);

    std::string output_wspecifier;
    if (!cross_validate) {
      output_wspecifier = po.GetArg(6);
    }

    // read in CMVN statistics
    Matrix<double> cmvn_stats;
    if (ClassifyRspecifier(cmvn_rspecifier_or_rxfilename, NULL, NULL)
        != kNoRspecifier) {  // reading from a Table: per-speaker or per-utt CMN/CVN.
      std::string cmvn_rspecifier = cmvn_rspecifier_or_rxfilename;

      RandomAccessDoubleMatrixReader cmvn_reader(cmvn_rspecifier);

      if (!cmvn_reader.HasKey("global")) {
        KALDI_ERR<< "No normalization statistics available for key "
        << "'global', producing no output for this utterance";
      }
        // read in the statistics
      cmvn_stats = cmvn_reader.Value("global");
    } else {  // read in the statistics in normal file format
      std::string cmvn_rxfilename = cmvn_rspecifier_or_rxfilename;
      bool binary;
      Input ki(cmvn_rxfilename, &binary);
      cmvn_stats.Read(ki.Stream(), binary);
    }

    //KALDI_LOG << "...cmvn stats:" << cmvn_stats;

    // generate the global mean and covariance from the statistics
    int32 feat_dim = cmvn_stats.NumCols() - 1;
    double counts = cmvn_stats(0, feat_dim);
    Vector<double> mean(feat_dim, kSetZero), var(feat_dim, kSetZero);
    for (int32 i = 0; i < feat_dim; ++i) {
      mean(i) = cmvn_stats(0, i) / counts;
      var(i) = (cmvn_stats(1, i) / counts) - mean(i) * mean(i);
    }

    KALDI_LOG<< "...update flag="<< update_flag;
    if (update_flag == "cmvn") {
      KALDI_LOG<< "Before update: \nMean: " << mean << "\n Var: " << var;
    }

      // currently no feature transform is supported
    Nnet nnet_transf;

    Nnet nnet;
    nnet.Read(nnet_filename);


    // only allow the first layer, which is <cmvnbl> to be updated
    std::string learn_factors = "1";
    for (int32 i = 1; i < nnet.LayerCount(); ++i) {
      if ((nnet.Layer(i))->IsUpdatable()) {
        learn_factors += ",0";
      }
    }
    KALDI_LOG<< "...learn_factor=" << learn_factors;
    nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);


    if (nnet.Layer(0)->GetType() != Component::kCMVNBL) {
      KALDI_ERR<< "The first layer is not <cmvnbl> layer!";
    }
    CMVNBL *cmvnbl_layer = (CMVNBL*) (nnet.Layer(0));

    // initialize the cmvnbl layer
    if (!cross_validate) {
      cmvnbl_layer->SetUpdateFlag(update_flag, update_vars);
    }
    cmvnbl_layer->SetParamKind(have_energy, num_fbank, delta_order);
    cmvnbl_layer->SetCMVN(mean, var);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    // only used when update_flag = "noise"
    DoubleVectorWriter noise_writer;
    if (!cross_validate && update_flag == "noise") {
      noise_writer.Open(output_wspecifier);
    }

    Xent xent;

    CuMatrix<BaseFloat> feats, nnet_out, glob_err;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (cross_validate? "CV":("TRAINING "+update_flag)) << " STARTED";

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

        cmvnbl_layer->SetNoise(mu_h, mu_z, var_z);

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

        nnet.Propagate(feats, &nnet_out);

        xent.EvalVec(nnet_out, alignment, &glob_err);

        if (!cross_validate) {
          nnet.Backpropagate(glob_err, NULL);
        }

        if(!cross_validate && update_flag=="noise") {
          cmvnbl_layer->GetNoise(mu_h, mu_z, var_z);
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

    if (!cross_validate && update_flag == "cmvn") {
      cmvnbl_layer->GetCMVN(mean, var);
      cmvn_stats.SetZero();
      cmvn_stats(0, feat_dim) = 1.0;
      for (int32 i = 0; i < feat_dim; ++i) {
        cmvn_stats(0, i) = mean(i);
        cmvn_stats(1, i) = var(i) + mean(i) * mean(i);
      }
      DoubleMatrixWriter cmvn_writer(output_wspecifier);
      cmvn_writer.Write("global", cmvn_stats);

      KALDI_LOG<< "After update: \nMean: " << mean << "\n Var: " << var;
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
