/*
 * analyze-gmm-nnet.cc
 *
 *  Created on: Sep 11, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Analyze the classification effectiveness of NNet input layers on the GMM Gaussians.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"
#include "gmm/diag-gmm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-biasedlinearity.h"
#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute the 1st NNet layer activations for GMM Gaussians (Row is GMM, column is NN node.\n"
            "Usage:  analyze-gmm-nnet  [options] <gmm-model-in> <nnet-model-in> "
            " <out-filename> [<noiseparam-rspecifier>]\n"
            "e.g.: \n"
            " analyze-gmm-nnet gmm nnet act.txt ark:noise.ark\n";
    ParseOptions po(usage);

    bool apply_sigmoid = true;
    po.Register("apply-sigmoid", &apply_sigmoid, "Apply sigmoid nonlinearity to the activations.");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral,
                "Number of Cepstral components in MFCC.");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used in MFCC extraction.");

    int32 ceplifter = 22;
    po.Register("ceplifter", &ceplifter,
                "Cepstral lifting parameter for MFCC extraction.");

    po.Read(argc, argv);

    if (!(po.NumArgs() == 3 || po.NumArgs() == 4)) {
      po.PrintUsage();
      exit(1);
    }

    std::string gmm_filename = po.GetArg(1), /* GMM model */
    nnet_filename = po.GetArg(2), /* NNet */
    out_filename = po.GetArg(3), /* output result file */
    noiseparam_rspecifier = po.GetOptArg(4); /* Noise parameter archive */

    /*
     * Load In the GMM Model
     */
    DiagGmm gmm;
    {
      bool binary_read;
      Input ki(gmm_filename, &binary_read);
      gmm.Read(ki.Stream(), binary_read);
    }

    /*
     * Load in the NNet 1st layer
     */
    Nnet nnet;
    nnet.Read(nnet_filename);
    BiasedLinearity &layer = dynamic_cast<BiasedLinearity&>(*nnet.Layer(0));

    /*
     * Initialisation for VTS
     */
    if (noiseparam_rspecifier != "") {
      Matrix<double> dct_mat, inv_dct_mat;
      GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                        &inv_dct_mat);

      Vector<double> mu_h, mu_z, var_z;

      int32 flag = 0;
      SequentialDoubleVectorReader noise_reader(noiseparam_rspecifier);

      for (; flag != 7 && !noise_reader.Done(); noise_reader.Next()) {
        std::string key = noise_reader.Key();
        if ((flag & 1) == 0 && key.find("_mu_h") != std::string::npos) {
          mu_h=noise_reader.Value();
          flag = flag | 1;
        }
        if ((flag & 2) == 0 && key.find("_mu_z") != std::string::npos) {
          mu_z=noise_reader.Value();
          flag = flag | 2;
        }
        if ((flag & 4) == 0 && key.find("_var_z") != std::string::npos) {
          var_z=noise_reader.Value();
          flag = flag | 4;
        }
      }

      if (flag != 7) {
        KALDI_ERR<< "Cannot find all the noise parameters: mu_h, mu_z, var_z!";
      }

      // Compensate the GMMs
      std::vector<Matrix<double> > Jx(gmm.NumGauss()), Jz(gmm.NumGauss());
      CompensateDiagGmm(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat, inv_dct_mat, gmm, Jx, Jz);

    }

    /*
     * Treat GMM means as features
     */
    Matrix<BaseFloat> gmm_means(gmm.NumGauss(), gmm.Dim(), kSetZero);
    gmm.GetMeans(&gmm_means);

    // checking for NaN/inf
    for(int32 i=0; i < gmm_means.NumRows(); ++i) {
      for(int32 j=0; j < gmm_means.NumCols(); ++j){
        BaseFloat val = gmm_means(i,j);
        if (val != val){
          KALDI_ERR << "NaN in GMM means.";
        }
        if (val == std::numeric_limits<BaseFloat>::infinity()) {
          KALDI_ERR << "inf in GMM means.";
        }
      }
    }

    Matrix<BaseFloat> acts_host;
    CuMatrix<BaseFloat> feats, linacts, acts;
    feats.CopyFromMat(gmm_means);
    layer.Propagate(feats, &linacts);
    if (apply_sigmoid){
      acts.CopyFromMat(linacts);
      cu::Sigmoid(linacts, &acts);
      acts.CopyToMat(&acts_host);
    }else{
      linacts.CopyToMat(&acts_host);
    }

    // checking outputs for NaN/inf
    for(int32 i=0; i<acts_host.NumRows(); ++i){
      for(int32 j=0; j<acts_host.NumCols(); ++j){
        BaseFloat val = acts_host(i,j);
        if(val!=val){
          KALDI_ERR << "NaN in output acts.";
        }
        if(val == std::numeric_limits<BaseFloat>::infinity()){
          KALDI_ERR << "inf in output acts.";
        }
      }
    }

    /*
     * Write the output acts.
     */
    {
      std::ofstream ofs;
      ofs.open(out_filename.c_str());
      for(int32 i=0; i<acts_host.NumRows(); ++i){
        for(int32 j=0; j<acts_host.NumCols(); ++j){
          ofs << acts_host(i,j) << " ";
        }
        ofs << "\n";
      }
      ofs.close();
    }

    KALDI_LOG<< "Processed " << gmm.NumGauss() << " Gaussians with "
    << layer.OutputDim() << " hidden units.";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

