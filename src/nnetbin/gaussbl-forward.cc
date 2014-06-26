// nnetbin/gaussbl-forward.cc

// forward through the nnet with the GaussBL front layer

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-gaussbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
            "Usage: gaussbl-forward [options] <model-in> <feature-rspecifier> "
            "<feature-wspecifier> [<noise-rspecifier>]\n"
            "e.g.: \n"
            " gaussbl-forward nnet ark:features.ark ark:out.ark ark:noise.ark\n";

    ParseOptions po(usage);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts,
                "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register(
        "prior-scale",
        &prior_scale,
        "scaling factor of prior log-probabilites given by --class-frame-counts");

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register(
        "no-softmax",
        &no_softmax,
        "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    bool silent = false;
    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    bool have_noise = false;
    if (po.NumArgs() == 4) {
      have_noise = true;
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3),
        noise_rspecifier = po.GetOptArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    // the first layer of the nnet should be GaussBL layer
    Component *comp = nnet.Layer(0);
    KALDI_ASSERT(comp->GetType() == Component::kGaussBL);

    GaussBL *layer = static_cast<GaussBL*>(nnet.Layer(0));

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    // Read the class-counts, compute priors
    Vector<BaseFloat> tmp_priors;
    CuVector<BaseFloat> priors;
    if (class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();

      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0 / sum);
      if (apply_log || no_softmax) {
        tmp_priors.ApplyLog();
        tmp_priors.Scale(-prior_scale);
      } else {
        tmp_priors.ApplyPow(-prior_scale);
      }

      // push priors to GPU
      priors.CopyFromVec(tmp_priors);
    }

    Timer tim;
    if (!silent)
      KALDI_LOG<< "POSNEGBL FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in features of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in features of : " << key;
          }
        }

      if (have_noise) {
        // read noise parameters
        if (!noiseparams_reader.HasKey(key + "_mu_h")
            || !noiseparams_reader.HasKey(key + "_mu_z")
            || !noiseparams_reader.HasKey(key + "_var_z")) {
          KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        }
        Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
        Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
        Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));
        if (g_kaldi_verbose_level >= 1) {
          KALDI_LOG<< "Additive Noise Mean: " << mu_z;
          KALDI_LOG << "Additive Noise Covariance: " << var_z;
          KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
        }

        layer->SetNoise(compensate_var, mu_h, mu_z, var_z);
      }

      // push it to gpu
      feats.CopyFromMat(mat);
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);

      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }

      // divide posteriors by priors to get quasi-likelihoods
      if (class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out.MulColsVec(priors);
        }
      }

      //download from GPU
      nnet_out.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in NNet output of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in NNet coutput of : " << key;
          }
        }

      // write
      feature_writer.Write(key, nnet_out_host);

      // progress log
      if (num_done % 100 == 0) {
        if (!silent)
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += feats.NumRows();
    }

    // final message
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD FINISHED "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed();
    if (!silent)
      KALDI_LOG<< "Done " << num_done << " files";

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
