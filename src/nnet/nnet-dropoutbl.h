// nnet/nnet-dropoutbl.h

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

#ifndef KALDI_NNET_DROPOUTBL_H
#define KALDI_NNET_DROPOUTBL_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {

// Dropout version of biased linearity, for NN training only.
// The network needs to converted to standard biased linearity for testing.
class DropoutBL : public UpdatableComponent {
 public:
  DropoutBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : UpdatableComponent(dim_in, dim_out, nnet),
        linearity_(dim_out, dim_in),
        bias_(dim_out),
        linearity_corr_(dim_out, dim_in),
        bias_corr_(dim_out) {
  }
  ~DropoutBL() {
  }

  ComponentType GetType() const {
    return kBiasedLinearity;
  }

  void ReadData(std::istream &is, bool binary) {
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    if (prob_mask.NumCols() != in.NumCols()
        || prob_mask.NumRows() != in.NumRows()) {
      prob_mask.Resize(in.NumRows(), in.NumCols());
      prob_mask.Set(0.5);
      value_mask.Resize(in.NumRows(), in.NumCols());
    }
    cu_rand.BinarizeProbs(prob_mask, &value_mask);  // value mask is kept for use in back propagation
    prob_mask.CopyFromMat(value_mask);
    prob_mask.MulElements(in);  // prob_mask as masked input

    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, prob_mask, kNoTrans, linearity_, kTrans, 1.0);

  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // multiply error by weights
    out_err->AddMatMat(1.0, in_err, kNoTrans, linearity_, kNoTrans, 0.0);
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {
    // compute gradient
    prob_mask.CopyFromMat(value_mask);
    prob_mask.MulElements(input);  // only the active links are updated
    if (average_grad_) {
      linearity_corr_.AddMatMat(1.0 / input.NumRows(), err, kTrans, prob_mask,
                                kNoTrans,
                                momentum_);
      bias_corr_.AddRowSumMat(1.0 / input.NumRows(), err, momentum_);
    } else {
      linearity_corr_.AddMatMat(1.0, err, kTrans, prob_mask, kNoTrans,
                                momentum_);
      bias_corr_.AddRowSumMat(1.0, err, momentum_);
    }

    // l2 regularization
    if (l2_penalty_ != 0.0) {
      BaseFloat l2 = learn_rate_ * l2_penalty_ * input.NumRows();
      linearity_.AddMat(-l2, linearity_);
    }
    // l1 regularization
    if (l1_penalty_ != 0.0) {
      BaseFloat l1 = learn_rate_ * input.NumRows() * l1_penalty_;
      cu::RegularizeL1(&linearity_, &linearity_corr_, l1, learn_rate_);
    }
    // update
    linearity_.AddMat(-learn_rate_, linearity_corr_);
    bias_.AddVec(-learn_rate_, bias_corr_);
  }

  void WriteAsBiasedLinearity(std::ostream& os, bool binary) const {
    //header
    WriteToken(os, binary,
               Component::TypeToMarker(Component::kBiasedLinearity));
    WriteBasicType(os, binary, OutputDim());
    WriteBasicType(os, binary, InputDim());
    if (!binary)
      os << "\n";
    //data
    CuMatrix<BaseFloat> bl_linearity(linearity_.NumRows(),
                                     linearity_.NumCols());
    bl_linearity.Set(0.5);
    bl_linearity.MulElements(linearity_);
    bl_linearity.Write(os, binary);
    CuMatrix<BaseFloat> bl_bias(1, bias_.Dim());
    bl_bias.Set(0.5);
    bl_bias.MulColsVec(bias_);
    bl_bias.Write(os, binary);
  }

 private:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  CuRand<BaseFloat> cu_rand;  // random generator
  CuMatrix<BaseFloat> prob_mask;  // the dropout mask, input features are masked to dropout 50% before forward,
                                  // this mask is actually with all elements to be 0.5
  CuMatrix<BaseFloat> value_mask;  // elements are either 1 or 0
};

}  // namespace

#endif
