// nnetbin/nnet-train-xent-hardlab-perutt.cc

// Copyright 2011  Karel Vesely

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
#include "nnet/nnet-cache.h"
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
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
        "Usage:  nnet-train-xent-hardlab-frmshuff [options] <feature-rspecifier> <alignments-rspecifier> <rbm-in> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-xent-hardlab-perutt scp:train.scp ark:train.ali nnet.init nnet.iter1\n";

    ParseOptions po(usage);
    bool binary = false, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    // RBM options
    bool rbm_binarize = false,
        rbm_apply_log = false;
    po.Register("rbm-binarize", &rbm_binarize, "Binarize the RBM Bernoulli hidden activations");
    po.Register("rbm-apply-log", &rbm_apply_log, "Apply log to the RBM activations");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l2_upper_bound = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l2-upper-bound", &l2_upper_bound, "L2 upper bound (>=1.0)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad, "Whether to average the gradient in the bunch");

    std::string learn_factors = "";
    po.Register("learn-factors", &learn_factors, "Learning factor for each updatable layer, separated by ',' and work together with learn rate");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    std::string hidbias_rspecifier;
    po.Register("hidbias", &hidbias_rspecifier, "Hidden bias for each utterance");

    int32 bunchsize=512, cachesize=32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling");

    po.Read(argc, argv);

    if (po.NumArgs() != 5-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2),
        rbm_filename = po.GetArg(3),
        model_filename = po.GetArg(4);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(5);
    }

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet_rbm;
    nnet_rbm.Read(model_filename);
    KALDI_ASSERT(nnet_rbm.LayerCount()==1);
    KALDI_ASSERT(nnet_rbm.Layer(0)->GetType() == Component::kRbm);
    Rbm &rbm = dynamic_cast<Rbm&>(*nnet_rbm.Layer(0));

    Nnet nnet;
    nnet.Read(model_filename);

    if(learn_factors == ""){
      nnet.SetLearnRate(learn_rate, NULL);
    }else{
      nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    }
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL2UpperBound(l2_upper_bound);
    nnet.SetL1Penalty(l1_penalty);
    nnet.SetAverageGrad(average_grad);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessBaseFloatVectorReader hidbias_reader(hidbias_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    Cache cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, glob_err;
    std::vector<int32> targets;

    CuMatrix<BaseFloat> rbm_acts, rbm_acts_bin;
    CuRand<BaseFloat> cu_rand;

    Timer tim;
    double time_next=0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !feature_reader.Done()) {
        std::string key = feature_reader.Key();
        if (!alignments_reader.HasKey(key)) {
          num_no_alignment++;
        } else {
          // get feature alignment pair
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          const std::vector<int32> &alignment = alignments_reader.Value(key);
          // chech for dimension
          if ((int32)alignment.size() != mat.NumRows()) {
            KALDI_WARN << "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
            num_other_error++;
            continue;
          }

          if(hidbias_reader.IsOpen()){
            if(!hidbias_reader.HasKey(key)){
              KALDI_WARN << "Utterance " << key <<": Skipped because no hidbias found.";
              num_other_error++;
              continue;
            }
            rbm.SetHiddenBias(hidbias_reader.Value(key));
          }

          // push features to GPU
          feats.CopyFromMat(mat);
          // possibly apply transform
          nnet_transf.Feedforward(feats, &feats_transf);

          //forward through RBM
          rbm.Propagate(feats_transf, &rbm_acts);

          // alter the hidden values, so we can generate negative example
          if (rbm.HidType() == Rbm::BERNOULLI && rbm_binarize) {
            cu_rand.BinarizeProbs(rbm_acts, &rbm_acts_bin);
            rbm_acts.CopyFromMat(rbm_acts_bin);
          }

          if (rbm_apply_log) {
            rbm_acts.ApplyLog();
          }

          // add to cache
          cache.AddData(rbm_acts, alignment);
          num_done++;
        }
        Timer t_features;
        feature_reader.Next(); 
        time_next += t_features.Elapsed();
      }
      // randomize
      if (!crossvalidate && randomize) {
        cache.Randomize();
      }
      // report
      std::cerr << "Cache #" << ++num_cache << " "
                << (cache.Randomized()?"[RND]":"[NO-RND]")
                << " segments: " << num_done
                << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&nnet_in, &targets);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);
        xent.EvalVec(nnet_out, targets, &glob_err);
        if (!crossvalidate) {
          nnet.Backpropagate(glob_err, NULL);
        }
        tot_t += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done()) break;
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

    KALDI_LOG << xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
