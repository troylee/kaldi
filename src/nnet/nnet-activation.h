// nnet/nnet-activation.h

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

#ifndef KALDI_NNET_ACTIVATION_H
#define KALDI_NNET_ACTIVATION_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {

class Scale : public Component {
 public:
  Scale(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
     : Component(dim_in, dim_out, nnet), scale_(1.0){
  }

  ~Scale(){
  }

  ComponentType GetType() const {
    return kScale;
  }

  void SetScale(BaseFloat scale){
    scale_=scale;
  }

  void ReadData(std::istream &is, bool binary) {
    BaseFloat scale;

    ReadBasicType(is, binary, &scale);
    scale_=scale;
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, scale_);
    if (!binary) os << "\n";
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = s * x
    out->CopyFromMat(in);
    out->Scale(scale_);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err, CuMatrix<BaseFloat> *out_err) {
    // ey = s * ex
    out_err->CopyFromMat(in_err);
    out_err->Scale(scale_);
  }

 private:
  BaseFloat scale_;
};

class Dropout : public Component {
 public:
  Dropout(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
     : Component(dim_in, dim_out, nnet), drop_ratio_(0.5) {
  }

  ~Dropout(){
  }

  ComponentType GetType() const {
    return kDropout;
  }

  void SetDropRatio(BaseFloat ratio) {
    drop_ratio_ = ratio;
    if(prob_.NumRows() >0){
      prob_.Set(1-drop_ratio_);
    }
  }

  BaseFloat GetDropRatio() const {
   return drop_ratio_;
  }

  void ReadData(std::istream &is, bool binary) {
    BaseFloat ratio;

    ReadBasicType(is, binary, &ratio);
    if(ratio > 0.0 && ratio < 1.0){
      drop_ratio_=ratio;
    }else{
      KALDI_WARN << "Invalid drop ratio for the <dropout> layer, using default 0.5.";
    }
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, drop_ratio_);
    if (!binary) os << "\n";
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = mask .* x
    if(prob_.NumRows()!=in.NumRows() || prob_.NumCols() != in.NumCols()){
      prob_.Resize(in.NumRows(), in.NumCols());
      mask_.Resize(in.NumRows(), in.NumCols());
      prob_.Set(1-drop_ratio_);
    }
    cu_rand_.BinarizeProbs(prob_, &mask_);
    out->CopyFromMat(in);
    // switch off the masked units
    out->MulElements(mask_);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err, CuMatrix<BaseFloat> *out_err) {
    // ey = mask .* ex
    out_err->CopyFromMat(in_err);
    // use the same mask for the error derivatives
    out_err->MulElements(mask_);
  }

 private:
  BaseFloat drop_ratio_;

  CuRand<BaseFloat> cu_rand_;  // random generator
  CuMatrix<BaseFloat> prob_;  // matrix with values set to 1-drop_ratio,
                                  // random number less than this value will give a 1 mask value
  CuMatrix<BaseFloat> mask_;  // the dropout mask, elements are either 1 or 0

};

class Relu : public Component {
 public:
  Relu(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : Component(dim_in, dim_out, nnet) {
  }

  ~Relu(){
  }

  ComponentType GetType() const {
    return kRelu;
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = max(0, x)
    cu::Relu(in, out);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // ey = (y>0) * ex
    const CuMatrix<BaseFloat> &y = nnet_->PropagateBuffer()[nnet_->IndexOfLayer(*this) + 1];
    cu::DiffRelu(in_err, y, out_err);
  }
};

class SoftRelu : public Component {
 public:
  SoftRelu(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : Component(dim_in, dim_out, nnet) {
  }

  ~SoftRelu(){
  }

  ComponentType GetType() const {
    return kSoftRelu;
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = x>4.0 ? x : log(1 + e^x)
    // this is a piece wise implementation of log(1 + e^x)
    cu::SoftRelu(in, out);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // ey = (x>4.0) ? ex : (e^x / (1.0 + e^x))*ex
    const CuMatrix<BaseFloat> &x = nnet_->PropagateBuffer()[nnet_->IndexOfLayer(*this)];
    cu::DiffSoftRelu(in_err, x, out_err);
  }
};

class Sigmoid : public Component {
 public:
  Sigmoid(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : Component(dim_in, dim_out, nnet) {
  }
  ~Sigmoid() {
  }

  ComponentType GetType() const {
    return kSigmoid;
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    cu::Sigmoid(in, out);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // ey = y(1-y)ex
    const CuMatrix<BaseFloat> &y = nnet_->PropagateBuffer()[nnet_->IndexOfLayer(
        *this) + 1];
    cu::DiffSigmoid(in_err, y, out_err);
  }
};

class Softmax : public Component {
 public:
  Softmax(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : Component(dim_in, dim_out, nnet) {
  }
  ~Softmax() {
  }

  ComponentType GetType() const {
    return kSoftmax;
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    cu::Softmax(in, out);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // simply copy the error
    // (ie. assume crossentropy error function, 
    // while in_err contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    out_err->CopyFromMat(in_err);
  }
};

}  // namespace

#endif

