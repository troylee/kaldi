// nnetbin/lin-nnet-forward.cc

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network with LIN.\n"
            "Usage:  lin-nnet-forward [options] <model-in> <lin-weight-rsepcifier> <lin-bias-rspecifier> <feature-rspecifier> <feature-wspecifier>\n"
            "e.g.: \n"
            " lin-nnet-forward nnet ark:weight.ark ark:bias.ark ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    std::string utt2xform;
    po.Register("utt2xform", &utt2xform, "Utterance to LIN xform mapping");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts,
                "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register(
        "prior-scale",
        &prior_scale,
        "scaling factor of prior log-probabilites given by --class-frame-counts");

    bool apply_log = false, silent = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register(
        "no-softmax",
        &no_softmax,
        "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        weight_rspecifier = po.GetArg(2),
        bias_rspecifier = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        feature_wspecifier = po.GetArg(5);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    if (nnet.Layer(0)->GetType() != Component::kLinBL) {
      KALDI_ERR<< "The first layer is not <linbl> layer!";
    }
    LinBL *lin = static_cast<LinBL*>(nnet.Layer(0));

    kaldi::int64 tot_t = 0;

    RandomAccessTokenReader utt2xform_reader(utt2xform);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    RandomAccessBaseFloatMatrixReader weight_reader(weight_rspecifier);
    RandomAccessBaseFloatVectorReader bias_reader(bias_rspecifier);

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
      KALDI_LOG<< "MLP FEEDFORWARD STARTED";

    int32 num_done = 0;
    std::string cur_lin="", new_lin="";

    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();

      if(utt2xform==""){
        new_lin = key;
      }else{
        if(!utt2xform_reader.HasKey(key)){
          KALDI_ERR << "No mapping found for utterance " << key;
        }
        new_lin=utt2xform_reader.Value(key);
      }

      if(!weight_reader.HasKey(new_lin) || !bias_reader.HasKey(new_lin)){
        KALDI_ERR << "No LIN weight or bias for the input feature " << new_lin;
      }

      // update the LIN xform when necessary
      if(new_lin != cur_lin){
        const Matrix<BaseFloat> &weight = weight_reader.Value(new_lin);
        const Vector<BaseFloat> &bias = bias_reader.Value(new_lin);

        lin->SetLinearityWeight(weight, false);
        lin->SetBiasWeight(bias);

        cur_lin = new_lin;
      }

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
      if (num_done % 1000 == 0) {
        if (!silent)
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD FINISHED "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed();
    if (!silent)
      KALDI_LOG<< "Done " << num_done << " files";

#if HAVE_CUDA==1
      if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
