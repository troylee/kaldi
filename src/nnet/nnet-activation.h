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

namespace kaldi {

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

