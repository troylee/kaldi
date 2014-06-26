// nnetbin/nnet-train-xent-hardlab-perutt.cc

// Copyright 2014  Bo Li 

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

/*
This program takes in parallel data and use the cross entropy for the primary data 
as the main objective of learning, and at each specified hidden layers, the mean 
sqaure errors as the second learning objective. 
*/


#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-xent-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include <sstream>


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
        "Usage:  nnet-train-stereo [options] <model-in> <noisyfeat-rspecifier> \n"
        "          <cleanfeat-rspecifier> <alignments-rspecifier> <model-out>\n"
        "e.g.: \n"
        " nnet-train-stereo --num-regularized-hid=1 nnet.init scp:train_noisy.scp scp:train_clean.scp ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);

    //=====================
    // Common options
    //
    bool binary = false, 
        crossvalidate = false,
        randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad, "Whether to average the gradient in the bunch");

    // not supported yet, all the layers inside each nnet share the same learning rate factor.
    //std::string learn_factors = "";
    //po.Register("learn-factors", &learn_factors, "Learning factor for each updatable layer, separated by ',' and work together with learn rate");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    int32 bunchsize=512, cachesize=32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling");

    //=====================
    // New options
    //
    int32 num_regularized_hid=0;
    po.Register("num-regularized-hid", &num_regularized_hid, "Number of hidden layers to be regularized by stereo data");

    BaseFloat diff_scaling = 1.0;
    po.Register("diff-scaling", &diff_scaling, "Scale factor for the differences");

    // process options
    po.Read(argc, argv);

    if (po.NumArgs() != 5 - (crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    if (num_regularized_hid<=0) {
      KALDI_ERR << "No hidden layers to regularize, choose a proper training tool!";
    }

    std::string model_filename = po.GetArg(1),
        noisyfeat_rspecifier = po.GetArg(2),
        cleanfeat_rspecifier = po.GetArg(3),
        alignments_rspecifier = po.GetArg(4),
        target_model_filename;

    if(!crossvalidate) {
      target_model_filename = po.GetArg(5);
    } 
     
    using namespace kaldi;
    typedef kaldi::int32 int32;


    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    //=====================
    // the assumed mode files are:
    // model_filename.0, model_filename.1, ..., model_filename.(n-1), model_filename.n
    //
    // Totally, there will be num_regularized_hid+1 nnets, 
    // the extra one is the output layer
    std::vector<Nnet*> layers;
    layers.resize(num_regularized_hid);
    for(int32 i=0; i<=num_regularized_hid; ++i){
      std::stringstream ss;
      ss << i;

      layers[i]=new Nnet();

      layers[i]->Read(model_filename+"."+ss.str());
      if(g_kaldi_verbose_level > 1) {
        KALDI_LOG << "Loaded layer: " << model_filename+"."+ss.str();
        KALDI_LOG << "Layer " << i << " count: " << layers[i]->LayerCount();
      }

      // learning configurations
      layers[i]->SetLearnRate(learn_rate, NULL);
      layers[i]->SetMomentum(momentum);
      layers[i]->SetL2Penalty(l2_penalty);
      layers[i]->SetL1Penalty(l1_penalty);
      layers[i]->SetAverageGrad(average_grad);
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader noisyfeat_reader(noisyfeat_rspecifier);
    SequentialBaseFloatMatrixReader cleanfeat_reader(cleanfeat_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    CacheXentTgtMat cache;
    cachesize = (cachesize/bunchsize)*bunchsize; // ensure divisibility
    cache.Init(cachesize, bunchsize);

    // only one cross entropy criterion
    Xent xent;
    // multiple MSE criterion
    std::vector<Mse> mse(num_regularized_hid);
    
    CuMatrix<BaseFloat> noisyfeats, cleanfeats, noisyfeats_transf, cleanfeats_transf, nnet_in_noisy, nnet_in_clean, nnet_out, glob_err;
    std::vector<int32> targets;

    // hidden activations
    std::vector<CuMatrix<BaseFloat> > hid_acts_noisy, hid_acts_clean, hid_err, backward_err;

    // explicitly initialize the CuMatrix vectors
    hid_acts_noisy.resize(num_regularized_hid);
    hid_acts_clean.resize(num_regularized_hid);
    hid_err.resize(num_regularized_hid);
    backward_err.resize(num_regularized_hid);

    Timer tim;
    double time_next=0;
    KALDI_LOG << (crossvalidate? "CV" : "TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !noisyfeat_reader.Done() && !cleanfeat_reader.Done()) {
        // read in the keys 
        std::string noisykey = noisyfeat_reader.Key();
        std::string cleankey = cleanfeat_reader.Key();

        // ensure the keys are matching
        if (noisykey != cleankey) {
          KALDI_ERR << "The two features must in parallel! Got [" << noisykey << "] vs. [" << cleankey <<"].";
        }

        // now check the alignment
        if (!alignments_reader.HasKey(noisykey)) {
          num_no_alignment++;
        } else {
          // get feature alignment pair
          const Matrix<BaseFloat> &noisymat = noisyfeat_reader.Value();
          const Matrix<BaseFloat> &cleanmat = cleanfeat_reader.Value();
          const std::vector<int32> &alignment = alignments_reader.Value(noisykey);

          // chech for dimension
          if(noisymat.NumRows() != cleanmat.NumRows() || noisymat.NumCols() != cleanmat.NumCols()) {
            KALDI_WARN << "Feature dimension mismatches! Feature 1: [" << noisymat.NumRows() << ", " << noisymat.NumCols() << "], while feature 2: [" << cleanmat.NumRows() << ", " << cleanmat.NumCols() << "].";
            num_other_error++;
            continue;
          }
          // check for alignment size
          if ((int32)alignment.size() != noisymat.NumRows()) {
            KALDI_WARN << "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (noisymat.NumRows());
            num_other_error++;
            continue;
          }

          // push features to GPU
          noisyfeats.CopyFromMat(noisymat);
          cleanfeats.CopyFromMat(cleanmat);
          // possibly apply transform
          nnet_transf.Feedforward(noisyfeats, &noisyfeats_transf);
          nnet_transf.Feedforward(cleanfeats, &cleanfeats_transf);
          // add to cache
          cache.AddData(noisyfeats_transf, alignment, cleanfeats_transf);
          num_done++;
        }
        Timer t_features;
        noisyfeat_reader.Next(); 
        cleanfeat_reader.Next();
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
        cache.GetBunch(&nnet_in_noisy, &targets, &nnet_in_clean);
        //==============================
        // forward through hidden layers
        for (int32 i=0; i<num_regularized_hid; ++i) {
          // forward propagation
          if (i==0) {
            layers[i]->Propagate(nnet_in_noisy, &hid_acts_noisy[i]);
            layers[i]->Propagate(nnet_in_clean, &hid_acts_clean[i]);
          }else{
            layers[i]->Propagate(hid_acts_noisy[i-1], &hid_acts_noisy[i]);
            layers[i]->Propagate(hid_acts_clean[i-1], &hid_acts_clean[i]);
          }
          // evaluate the MSE 
          mse[i].Eval(hid_acts_noisy[i], hid_acts_clean[i], &hid_err[i]);
        }
        //===============================
        // forward through the last layer
        int32 lastID=num_regularized_hid;
        layers[lastID]->Propagate(hid_acts_noisy[lastID-1], &nnet_out);
        // evaluate the Xent
        xent.EvalVec(nnet_out, targets, &glob_err);
        if(!crossvalidate) {
          //================================
          // backpropagate the last layer first
          layers[lastID]->Backpropagate(glob_err, &backward_err[lastID-1]);
          //================================
          // backpropagate hidden layers
          for (int32 i=num_regularized_hid-1; i>=0; --i) {
            // adding the hidden errors
            backward_err[i].AddMat(diff_scaling, hid_err[i], 1.0);
            // backpropagate
            if (i>0) {
              layers[i]->Backpropagate(backward_err[i], &backward_err[i-1]);          
            } else {
              layers[i]->Backpropagate(backward_err[i], NULL);
            }
          }
        }

        tot_t += nnet_in_noisy.NumRows();
      }

      // stop training when no more data
      if (noisyfeat_reader.Done() || cleanfeat_reader.Done()) break;
    }

    if(!crossvalidate) {

      // write out the model
      for (int32 i=0; i<num_regularized_hid+1; ++i) {
        std::stringstream ss;
        ss << i;
        layers[i]->Write(target_model_filename + "." +ss.str(), binary);
        if(g_kaldi_verbose_level > 1) {
          KALDI_LOG << "Saved layer: " << model_filename << "." << i;
        }
      }
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << (crossvalidate? "CV " : "TRAINING ") << "FINISHED " 
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

    KALDI_LOG << xent.Report();

    KALDI_LOG << "Hidden layer stats:";
    for(int32 i=0; i<num_regularized_hid; ++i) {
      KALDI_LOG << "Layer " << i << ":";
      KALDI_LOG << mse[i].Report();
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
