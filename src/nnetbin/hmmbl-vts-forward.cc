/*
 * hmmbl-vts-forward.cc
 *
 *  Created on: Sep 12, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Forward features through <hmmbl> layer with VTS compensation.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-hmmbl.h"
#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try{
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Forward features through the <hmmbl> layer.\n"
        "Usage: hmmbl-vts-forward [options] <nnet-model-in> <feats-rspecifier>"
        "<outs-wspecifier> [<noiseparam-rspecifier>]\n"
        "e.g.\n"
        "hmmbl-vts-forward nnet ark:feats.ark ark:acts.ark ark:noise.ark\n";

    ParseOptions po(usage);

    bool apply_exp = true;
    po.Register("apply-exp", &apply_exp, "Apply Exponential to the acts, such that the acts are exactly the likelihood.");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral, "Number of cepstral features in MFCC.");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank, "Number of FBanks in MFCC feature extraction.");

    int32 ceplifter = 22;
    po.Register("ceplifter", &ceplifter, "Cepstral lifting parameter for MFCC.");

    po.Read(argc, argv);

    if(po.NumArgs()!=3 && po.NumArgs()!=4){
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_model_in = po.GetArg(1),
        feat_rspecifier = po.GetArg(2),
        out_wspecifier = po.GetArg(3),
        noiseparam_rspecifier = po.GetOptArg(4);

    Nnet nnet;
    nnet.Read(nnet_model_in);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kHMMBL);
    HMMBL &hmmbl = dynamic_cast<HMMBL&>(*nnet.Layer(0));

    hmmbl.EnableExp(apply_exp);

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat, &inv_dct_mat);

    std::vector<Matrix<double> > Jx(hmmbl.OutputDim()), Jz(hmmbl.OutputDim());

    kaldi::int64 tot_t = 0;
    int32 num_done = 0;

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessDoubleVectorReader noiseparam_reader(noiseparam_rspecifier);
    BaseFloatMatrixWriter out_writer(out_wspecifier);

    CuMatrix<BaseFloat> in, out;
    Matrix<BaseFloat> out_host;

    Timer tim;
    KALDI_LOG << "HMMBL FORWARD STARTED.";

    for(; !feat_reader.Done(); feat_reader.Next() ) {
      std::string key = feat_reader.Key();
      Matrix<BaseFloat> feat = feat_reader.Value();

      if(noiseparam_rspecifier!=""){
        // checking the existance of noise parameters
        if(!noiseparam_reader.HasKey(key+"_mu_h") || !noiseparam_reader.HasKey(key+"_mu_z") ||
            !noiseparam_reader.HasKey(key+"_var_z")){
          KALDI_ERR << "No noise parameter for utt: " << key << "!\n";
        }
        Vector<double> mu_h(noiseparam_reader.Value(key+"_mu_h"));
        Vector<double> mu_z(noiseparam_reader.Value(key+"_mu_z"));
        Vector<double> var_z(noiseparam_reader.Value(key+"_var_z"));

        hmmbl.VTSCompensate(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat, inv_dct_mat, Jx, Jz);
      }

      // checking features for NaN/inf
      for(int32 r=0; r<feat.NumRows(); ++r){
        for(int32 c=0; c<feat.NumCols(); ++c){
          BaseFloat val=feat(r,c);
          if(val!=val) KALDI_ERR << "NaN in features of " << key << ".\n";
          if(val== std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of " << key << ".\n";
        }
      }

      // add 2nd order features
      int32 num = feat.NumRows(), dim = feat.NumCols();
      feat.Resize(num, 2*dim, kCopyData);
      (SubMatrix<BaseFloat>(feat, 0, num, dim, dim)).CopyFromMat(SubMatrix<BaseFloat>(feat, 0, num, 0, dim));
      (SubMatrix<BaseFloat>(feat, 0, num, dim, dim)).ApplyPow(2.0);

      in.CopyFromMat(feat);
      hmmbl.Propagate(in, &out);

      out.CopyToMat(&out_host);

      // checking for NaN/inf
      for(int32 r=0; r<out_host.NumRows(); ++r){
        for(int32 c=0; c<out_host.NumCols(); ++c){
          BaseFloat val=out_host(r,c);
          if(val!=val) KALDI_ERR << "NaN in acts of " << key << ".\n";
          if(val==std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in acts of" << key << ".\n";
        }
      }

      out_writer.Write(key, out_host);
      ++num_done;
      tot_t += out_host.NumRows();

      if(num_done%1000==0){
        KALDI_LOG << "Processed " << num_done << " utterances.";
      }

    }

    KALDI_LOG << "HMMBL FORWARD FINISHED!";
    KALDI_LOG << "Totally processed " << num_done << " utterances. " << tim.Elapsed() << "s, fps" << tim.Elapsed()/tot_t;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

  } catch(const std::exception &e){
    std::cerr << e.what() << '\n';
    return -1;
  }
}



