// nnetbin/rbm-uttbias-forward.cc

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

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"
#include "nnet/nnet-rbm.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward propagation of RBM with utt-bias.\n"
        "Usage:  rbm-uttbias-forward [options] <model-in> <feature-rspecifier> <act-wspecifier>\n"
        "e.g.: \n"
        " rbm-uttbias-forward rbm.mdl --hidbias='ark:hidbias.ark' scp:train.scp ark:act.ark\n";

    ParseOptions po(usage);

    bool binarize = false;
    po.Register("binarize", &binarize, "Binarize the Bernoulli hidden activations");

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Apply log to the activations");

    int32 buffer_size = 1000;
    po.Register("buffer-size", &buffer_size, "The sample size used to pre-allocate memory");

    std::string feature_transform, hidbias_rspecifier;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");
    po.Register("hidbias", &hidbias_rspecifier, "Hidden bias for each utterance");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 ) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        act_wspecifier = po.GetArg(3);
     
    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet rbm_transf;
    if(feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRbm);
    Rbm &rbm = dynamic_cast<Rbm&>(*nnet.Layer(0));

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessBaseFloatVectorReader hidbias_reader(hidbias_rspecifier);
    BaseFloatMatrixWriter act_writer(act_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, acts, acts_bin;

    CuRand<BaseFloat> cu_rand;

    Matrix<BaseFloat> expanded_mat(buffer_size, rbm.InputDim()), expanded_acts_host, acts_host;

    int32 zero_ro, zero_r;

    Timer tim;
    double time_next=0;
    KALDI_LOG << "RBM FORWARD PROPAGATION STARTED";

    int32 num_done = 0, num_other_error = 0;
    for (; !feature_reader.Done(); /*feature_reader.Next()*/) {
      std::string key = feature_reader.Key();
      KALDI_VLOG(3) << key;

      if(hidbias_reader.IsOpen()){
        if(!hidbias_reader.HasKey(key)){
          KALDI_WARN << "Utterance " << key <<": Skipped because no hidbias found.";
          num_other_error++;
          continue;
        }
        rbm.SetHiddenBias(hidbias_reader.Value(key));
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
      //check for NaN/inf
      for (int32 r = 0; r<mat.NumRows(); r++) {
        for (int32 c = 0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          if (val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if (val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }

      /*
       * To avoid too frequently allocating and freeing the GPU memory, which may lead to
       * GPU memory overflow. That is due to the memory is not instantly freed after calling
       * the free function.
       * We hence maintain the data size unless a larger one comes.
       */
      if(mat.NumRows() > expanded_mat.NumRows()){
        expanded_mat.Resize(mat.NumRows(), mat.NumCols(), kSetZero);
      }
      (SubMatrix<BaseFloat>(expanded_mat, 0, mat.NumRows(), 0, mat.NumCols())).CopyFromMat(mat);
      // keep track the row indices that are 0
      zero_ro = mat.NumRows(); // starting index of zero region
      zero_r = expanded_mat.NumRows() - mat.NumRows(); // total number of rows are zero

      // push features to GPU
      feats.CopyFromMat(expanded_mat);
      // possibly apply transforms
      rbm_transf.Feedforward(feats, &feats_transf);

      // forward pass
      rbm.Propagate(feats_transf, &acts);

      // alter the hidden values, so we can generate negative example
      if (rbm.HidType() == Rbm::BERNOULLI && binarize) {
        cu_rand.BinarizeProbs(acts, &acts_bin);
        acts.CopyFromMat(acts_bin);
      }

      if (apply_log) {
        acts.ApplyLog();
      }

      acts.CopyToMat(&expanded_acts_host);
      acts_host.CopyFromMat(SubMatrix<BaseFloat>(expanded_acts_host, zero_ro, zero_r, 0, expanded_acts_host.NumCols()));

      //check for NaN/inf
      for (int32 r = 0; r < acts_host.NumRows(); r++) {
        for (int32 c = 0; c < acts_host.NumCols(); c++) {
          BaseFloat val = acts_host(r,c);
          if (val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if (val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
        }
      }

      act_writer.Write(key, acts_host);

      tot_t += mat.NumRows();

      num_done++;
      if(num_done % 1000 == 0) std::cout << num_done << ", " << std::flush;

#if HAVE_CUDA==1
    KALDI_VLOG(9) << CuDevice::Instantiate().GetFreeMemory();
#endif
    
      Timer t_features;
      feature_reader.Next();
      time_next += t_features.Elapsed();
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << "RBM FORWARD PROPAGATION FINISHED "
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_other_error
              << " with other errors.";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
