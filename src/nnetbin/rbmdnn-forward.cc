// nnetbin/rbmdnn-forward.cc

// Copyright 2014 Bo Li

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-rand.h"
#include "nnet/nnet-rbm.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "Usage:  nnet-forward [options] <rbm-in> <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnet-forward rbm nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    std::string hidbias_rspecifier;
    po.Register("hidbias", &hidbias_rspecifier, "Hidden bias for each utterance");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts, "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register("prior-scale", &prior_scale, "scaling factor of prior log-probabilites given by --class-frame-counts");

    // RBM options
    bool rbm_binarize = false,
        rbm_apply_log = false;
    po.Register("rbm-binarize", &rbm_binarize, "Binarize the RBM Bernoulli hidden activations");
    po.Register("rbm-apply-log", &rbm_apply_log, "Apply log to the RBM activations");

    bool apply_log = false, silent = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string rbm_filename = po.GetArg(1),
        model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        feature_wspecifier = po.GetArg(4);
        
    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet_rbm;
    nnet_rbm.Read(rbm_filename);
    KALDI_ASSERT(nnet_rbm.LayerCount()==1);
    KALDI_ASSERT(nnet_rbm.Layer(0)->GetType() == Component::kRbm);
    Rbm &rbm = dynamic_cast<Rbm&>(*nnet_rbm.Layer(0));

    Nnet nnet;
    nnet.Read(model_filename);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessBaseFloatVectorReader hidbias_reader(hidbias_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    CuMatrix<BaseFloat> rbm_acts, rbm_acts_bin;
    CuRand<BaseFloat> cu_rand;

    // Read the class-counts, compute priors
    Vector<BaseFloat> tmp_priors;
    CuVector<BaseFloat> priors;
    if(class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();
      
      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0/sum);
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
    if(!silent) KALDI_LOG << "MLP FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      //check for NaN/inf
      for(int32 r=0; r<mat.NumRows(); r++) {
        for(int32 c=0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          if(val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }
      // configure RBM
      if(hidbias_reader.IsOpen()){
        if(!hidbias_reader.HasKey(key)){
          KALDI_ERR << "Utterance " << key <<" has no hidbias.";
        }
        rbm.SetHiddenBias(hidbias_reader.Value(key));
      }

      // push it to gpu
      feats.CopyFromMat(mat);
      // fwd-feature transform
      nnet_transf.Feedforward(feats, &feats_transf);

      // fwd-RBM
      rbm.Propagate(feats_transf, &rbm_acts);
      // alter the hidden values, so we can generate negative example
      if (rbm.HidType() == Rbm::BERNOULLI && rbm_binarize) {
        cu_rand.BinarizeProbs(rbm_acts, &rbm_acts_bin);
        rbm_acts.CopyFromMat(rbm_acts_bin);
      }
      if (rbm_apply_log) {
        rbm_acts.ApplyLog();
      }

      // fwd-nnet
      nnet.Feedforward(rbm_acts, &nnet_out);
      
      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }
     
      // divide posteriors by priors to get quasi-likelihoods
      if(class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out.MulColsVec(priors);
        }
      }
     
      //download from GPU 
      nnet_out.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for(int32 r=0; r<nnet_out_host.NumRows(); r++) {
        for(int32 c=0; c<nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r,c);
          if(val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
          /*if(val == kBaseLogZero)
            nnet_out_host(r,c) = -1e10;*/
        }
      }
      // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 1000 == 0) {
        if(!silent) KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    if(!silent) KALDI_LOG << "MLP FEEDFORWARD FINISHED " 
                          << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed(); 
    if(!silent) KALDI_LOG << "Done " << num_done << " files";

#if HAVE_CUDA==1
    if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done>0)?0:1);
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
