// nnetbin/rbm-uttbias-train.cc

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
        "Perform iteration of RBM training by CD1.\n"
        "Usage:  rbm-uttbias-train [options] <feature-rspecifier> "
        "<visbias-wspecifier> <hidbias-wspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " rbm-uttbias-train scp:train.scp ark:visbias.0.ark ark:visbias.1.ark"
        "ark:hidbias.0.ark ark:hidbias.1.ark rbm.init rbm.iter1\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");

    std::string feature_transform,
      visbias_rspecifier, hidbias_rspecifier;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");
    po.Register("init-visbias", &visbias_rspecifier, "Initial visible biases");
    po.Register("init-hidbias", &hidbias_rspecifier, "Initial hidden biases");

    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        visbias_wspecifier = po.GetArg(2),
        hidbias_wspecifier = po.GetArg(3),
        model_filename = po.GetArg(4),
        target_model_filename = po.GetOptArg(5);
     
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

    rbm.SetLearnRate(learn_rate);
    rbm.SetMomentum(momentum);
    rbm.SetL2Penalty(l2_penalty);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    RandomAccessBaseFloatVectorReader visbias_reader, hidbias_reader;
    if(visbias_rspecifier != ""){
      visbias_reader.Open(visbias_rspecifier);
    }
    if(hidbias_rspecifier != ""){
      hidbias_reader.Open(hidbias_rspecifier);
    }

    BaseFloatVectorWriter visbias_writer(visbias_wspecifier);
    BaseFloatVectorWriter hidbias_writer(hidbias_wspecifier);

    MseProgress mse;

    CuMatrix<BaseFloat> feats, feats_transf, pos_vis, pos_hid, neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;

    // biases
    Vector<BaseFloat> visbias(rbm.InputDim(), kSetZero),
        global_visbias(rbm.InputDim(), kSetZero),
        hidbias(rbm.OutputDim(), kSetZero),
        global_hidbias(rbm.OutputDim(), kSetZero);

    // keep a copy of original RBM parameters
    Vector<BaseFloat> init_visbias(rbm.InputDim(), kSetZero),
        init_hidbias(rbm.OutputDim(), kSetZero);
    Matrix<BaseFloat> init_weight(rbm.OutputDim(), rbm.InputDim());
    rbm.GetWeight(&init_weight);
    rbm.GetVisibleBias(&init_visbias);
    rbm.GetHiddenBias(&init_hidbias);

    Timer tim;
    double time_next=0;
    KALDI_LOG << "RBM TRAINING STARTED";

    int32 num_done = 0, num_other_error = 0;
    for (; !feature_reader.Done(); /*feature_reader.Next()*/) {
      std::string key = feature_reader.Key();
      KALDI_VLOG(3) << key;

      // setup the RBM model
      if(visbias_reader.IsOpen()){
        if(!visbias_reader.HasKey(key)){
          KALDI_WARN << "Utterance " << key <<": Skipped because no visbias found.";
          num_other_error++;
          continue;
        }
        visbias.CopyFromVec(visbias_reader.Value(key));
      }else{
        visbias.CopyFromVec(init_visbias);
      }
      if(hidbias_reader.IsOpen()){
        if(!hidbias_reader.HasKey(key)){
          KALDI_WARN << "Utterance " << key <<": Skipped because no visbias found.";
          num_other_error++;
          continue;
        }
        hidbias.CopyFromVec(hidbias_reader.Value(key));
      }else{
        hidbias.CopyFromVec(init_hidbias);
      }
      rbm.SetVisibleBias(visbias);
      rbm.SetHiddenBias(hidbias);

      // only learn biases, use the same weight for all the utts
      if(target_model_filename == "") {
        rbm.SetWeight(init_weight);
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
      CuRand<BaseFloat> cu_rand;

      // push features to GPU
      feats.CopyFromMat(mat);
      // possibly apply transforms
      rbm_transf.Feedforward(feats, &pos_vis);

      // TRAIN with CD1
      // forward pass
      rbm.Propagate(pos_vis, &pos_hid);
      // alter the hidden values, so we can generate negative example
      if (rbm.HidType() == Rbm::BERNOULLI) {
        cu_rand.BinarizeProbs(pos_hid, &neg_hid);
      } else {
        // assume Rbm::GAUSSIAN
        neg_hid.CopyFromMat(pos_hid);
        cu_rand.AddGaussNoise(&neg_hid);
      }
      // reconstruct pass
      rbm.Reconstruct(neg_hid, &neg_vis);
      // propagate negative examples
      rbm.Propagate(neg_vis, &neg_hid);
      // update step
      rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
      // evaluate mean square error
      mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

      // write out the new estimates
      rbm.GetVisibleBias(&visbias);
      rbm.GetHiddenBias(&hidbias);
      visbias_writer.Write(key, visbias);
      hidbias_writer.Write(key, hidbias);

      if(target_model_filename != "") {
        // accumulate global stats for updating the model
        global_visbias.AddVec(1.0, visbias);
        global_hidbias.AddVec(1.0, hidbias);
      }

      tot_t += pos_vis.NumRows();

      num_done++;
      if(num_done % 1000 == 0) std::cout << num_done << ", " << std::flush;
    
      Timer t_features;
      feature_reader.Next();
      time_next += t_features.Elapsed();
    }

    if (target_model_filename != "") {
      global_visbias.Scale(1.0/num_done);
      global_hidbias.Scale(1.0/num_done);

      rbm.SetVisibleBias(global_visbias);
      rbm.SetHiddenBias(global_hidbias);

      nnet.Write(target_model_filename, binary);
    }
    
    std::cout << "\n" << std::flush;

    KALDI_LOG << "RBM TRAINING FINISHED "
              << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_other_error
              << " with other errors.";

    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
