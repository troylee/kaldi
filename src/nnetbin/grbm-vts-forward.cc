// nnetbin/grbm-vts-forward.cc

/*
 * Created on: Sept. 10, 2013
 *     Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *     Forward input features through VTS compensated GB-RBM.
 */

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-grbm.h"
#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {

  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Perform forward pass through GBRBM with VTS compensation.\n"
            "Usage:  grbm-vts-forward [options] <model-in> <feature-rspecifier> <noiseparam-rspecifier>"
            " <feature-wspecifier>\n"
            "e.g.: \n"
            " grbm-vts-forward nnet ark:features.ark ark:noiseparams.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    bool silent = false;
    po.Register("silent", &silent, "Don't print any messages");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");

    BaseFloat ceplifter = 22;
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        noiseparam_rspecifier = po.GetArg(3),
        feature_wspecifier = po.GetArg(4);

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kGRbm);
    GRbm &grbm = dynamic_cast<GRbm&>(*nnet.Layer(0));

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noiseparam_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    Timer tim;
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD STARTED";

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
            KALDI_ERR<< "NaN in features of : " << feature_reader.Key();
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in features of : " << feature_reader.Key();
          }
        }

            // read noise parameters
      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR
            << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
          }
      Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
      Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
      Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));
      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG<< "Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
      }

      grbm.VTSInit();
      grbm.VTSCompensate(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat,
                         inv_dct_mat);

      // push it to gpu
      feats.CopyFromMat(mat);
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);

      grbm.VTSClear();

      //download from GPU 
      nnet_out.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in NNet output of : " << feature_reader.Key();
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in NNet coutput of : " << feature_reader.Key();
          }
        }
            // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

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
