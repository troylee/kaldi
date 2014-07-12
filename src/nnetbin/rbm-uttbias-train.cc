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
        "Usage:  rbm-uttbias-train [options] <feature-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " rbm-uttbias-train --visbias-in='ark:visbias.0.ark' --visbias-out='ark:visbias.1.ark'"
        " --hidbias-in='ark:hidbias.0.ark' --hidbias-out='ark:hidbias.1.ark' scp:train.scp rbm.init rbm.iter1\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");

    std::string feature_transform = "",
      visbias_rspecifier = "", visbias_wspecifier = "",
      hidbias_rspecifier = "", hidbias_wspecifier = "";
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");
    po.Register("visbias-in", &visbias_rspecifier, "Input visible biases");
    po.Register("visbias-out", &visbias_wspecifier, "Output visible biases");
    po.Register("hidbias-in", &hidbias_rspecifier, "Input hidden biases");
    po.Register("hidbias-out", &hidbias_wspecifier, "Output hidden biases");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        model_filename = po.GetArg(2),
        target_model_filename = po.GetOptArg(3);
     
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
    Rbm &rbm = dynamic_cast<Rbm&>(*(nnet.Layer(0)));

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

    BaseFloatVectorWriter visbias_writer, hidbias_writer;
    if(visbias_wspecifier != ""){
      visbias_writer.Open(visbias_wspecifier);
    }
    if(hidbias_wspecifier != ""){
      hidbias_writer.Open(hidbias_wspecifier);
    }

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

    MseProgress mse;

    CuMatrix<BaseFloat> feats, feats_transf, pos_vis, pos_hid, neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;

    Matrix<BaseFloat> expanded_mat;
    int32 zero_ro, zero_r;
    int32 dim_vis = rbm.InputDim(), dim_hid = rbm.OutputDim();

    Timer tim;
    double time_next=0;
    KALDI_LOG << "RBM TRAINING STARTED";

    int32 num_done = 0, num_other_error = 0;
    for (; !feature_reader.Done(); /*feature_reader.Next()*/) {
      std::string key = feature_reader.Key();
      KALDI_VLOG(3) << key;

      /*******************************************
       * Reset the RBM weights when necessary
       *******************************************/
      /* setup the RBM visible bias */
      if(visbias_reader.IsOpen()){ // using specified per-utt bias
        if(!visbias_reader.HasKey(key)){
          KALDI_WARN << "Utterance " << key <<": Skipped because no visbias found.";
          num_other_error++;
          continue;
        }
        rbm.SetVisibleBias(visbias_reader.Value(key));
      }else if (visbias_writer.IsOpen() || target_model_filename == ""){ // resetting to the global initial value
        rbm.SetVisibleBias(init_visbias);
      } // otherwise, the bias is kept global

      /* setup the RBM hidden bias */
      if(hidbias_reader.IsOpen()){
        if(!hidbias_reader.HasKey(key)){ // using specified per-utt bias
          KALDI_WARN << "Utterance " << key <<": Skipped because no visbias found.";
          num_other_error++;
          continue;
        }
        rbm.SetHiddenBias(hidbias_reader.Value(key));
      }else if (hidbias_writer.IsOpen() || target_model_filename == ""){ // resetting to the global initial value
        rbm.SetHiddenBias(init_hidbias);
      } // otherwise, the bias is kept global

      /* setup the RBM weight */
      if(target_model_filename == "") {
        rbm.SetWeight(init_weight);
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
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

      CuRand<BaseFloat> cu_rand;
      KALDI_VLOG(3) << "Feature size: [" << mat.NumRows() << ", " << mat.NumCols() << "]";

      // push features to GPU
      feats.CopyFromMat(expanded_mat);
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

      // reset zero regions
      if(zero_r > 0){
        pos_vis.PartSet(0.0, zero_ro, zero_r, 0, dim_vis);
        pos_hid.PartSet(0.0, zero_ro, zero_r, 0, dim_hid);
        neg_vis.PartSet(0.0, zero_ro, zero_r, 0, dim_vis);
        neg_hid.PartSet(0,0, zero_ro, zero_r, 0, dim_hid);
      }
      // update step
      rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
      // evaluate mean square error
      mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

      // write out the new estimates
      if(visbias_writer.IsOpen()){
        rbm.GetVisibleBias(&visbias);
        visbias_writer.Write(key, visbias);
      }
      if(hidbias_writer.IsOpen()){
        rbm.GetHiddenBias(&hidbias);
        hidbias_writer.Write(key, hidbias);
      }
      if(target_model_filename != "") {
        // accumulate global stats for biases
        global_visbias.AddVec(1.0, visbias);
        global_hidbias.AddVec(1.0, hidbias);
      }

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

    if (target_model_filename != "") {
      global_visbias.Scale(1.0/num_done);
      global_hidbias.Scale(1.0/num_done);

      /* We use the average bias as the dummy value in the RBM model file
       * for utt-biases. */
      if(visbias_writer.IsOpen()){
        rbm.SetVisibleBias(global_visbias);
      }
      if(hidbias_writer.IsOpen()){
        rbm.SetHiddenBias(global_hidbias);
      }

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
