/*
 * nnet-codebl.h
 *
 *  Created on: Oct 2, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      This kind of BL layer has a code input, which may be representing speaker,
 *      environment, etc factors. This code vector is augmented with the input to be
 *      forwarded.
 *
 *      Assume the code is before the input.
 */

#ifndef NNET_CODEBL_H_
#define NNET_CODEBL_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-biasedlinearity.h"

namespace kaldi {

class CodeBL : public BiasedLinearity {
 public:
  CodeBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : BiasedLinearity(dim_in, dim_out, nnet),
        update_weight_(true),
        code_dim_(0)

  {

  }

  ~CodeBL() {
  }

  ComponentType GetType() const {
    return kCodeBL;
  }

  /*
   * The reason to overwrite this read function is the matrix size won't match
   * with the input feature dim due to the additional code vector.
   */
  void ReadData(std::istream &is, bool binary) {
    ReadBasicType(is, binary, &code_dim_);
    KALDI_ASSERT(code_dim_ >= 0);
    code_vec_.Resize(code_dim_);
    code_corr_.Resize(code_dim_);

    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_ + code_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);

    linearity_corr_.Resize(linearity_.NumRows(), linearity_.NumCols());
    bias_corr_.Resize(bias_.Dim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, code_dim_);
    if (!binary)
      os << "\n";

    linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // augment the input if needed
    if (code_dim_ > 0) {
      if (augmented_in_.NumRows() != in.NumRows()
          || augmented_in_.NumCols() != in.NumCols() + code_dim_) {
        augmented_in_.Resize(in.NumRows(), in.NumCols() + code_dim_);
      }
      augmented_in_.AddVecToPartialRows(1.0, 0, code_vec_, 0.0);
      augmented_in_.PartAddMat(1.0, in, 0, code_dim_, 0.0);
    } else {
      augmented_in_.CopyFromMat(in);
    }

    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, augmented_in_, kNoTrans, linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // multiply error by weights
    if (augmented_err_.NumRows() != in_err.NumRows()
        || augmented_err_.NumCols() != linearity_.NumCols()) {
      augmented_err_.Resize(in_err.NumRows(), linearity_.NumCols());
    }
    augmented_err_.AddMatMat(1.0, in_err, kNoTrans, linearity_, kNoTrans, 0.0);

    if (code_dim_ > 0) {
      code_err_.CopyFromMat(augmented_err_, 0, augmented_err_.NumRows(), 0,
                            code_dim_);
      out_err->CopyFromMat(augmented_err_, 0, augmented_err_.NumRows(),
                           code_dim_, augmented_err_.NumCols() - code_dim_);
    } else {
      out_err->CopyFromMat(augmented_err_);
    }
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {
    // compute gradient
    if (average_grad_) {
      if (update_weight_) {
        linearity_corr_.AddMatMat(1.0 / input.NumRows(), err, kTrans,
                                  augmented_in_, kNoTrans, momentum_);
        bias_corr_.AddRowSumMat(1.0 / input.NumRows(), err, momentum_);
      }
      code_corr_.AddRowSumMat(1.0 / input.NumRows(), code_err_, momentum_);
    } else {
      if (update_weight_) {
        linearity_corr_.AddMatMat(1.0, err, kTrans, augmented_in_, kNoTrans,
                                  momentum_);
        bias_corr_.AddRowSumMat(1.0, err, momentum_);
      }
      code_corr_.AddRowSumMat(1.0, code_err_, momentum_);
    }
    // l2 regularization
    if (l2_penalty_ != 0.0 && update_weight_) {
      BaseFloat l2 = learn_rate_ * l2_penalty_ * input.NumRows();
      linearity_.AddMat(-l2, linearity_);
    }
    // l1 regularization
    if (l1_penalty_ != 0.0 && update_weight_) {
      BaseFloat l1 = learn_rate_ * input.NumRows() * l1_penalty_;
      cu::RegularizeL1(&linearity_, &linearity_corr_, l1, learn_rate_);
    }
    // update
    if (update_weight_) {
      linearity_.AddMat(-learn_rate_, linearity_corr_);
      bias_.AddVec(-learn_rate_, bias_corr_);
    }

    // scale the code_corr_ with current layer's learn rate
    code_corr_.Scale(-learn_rate_);

  }

  /*
   * The update of the code vector is done outside as it may be shared
   * among different layers.
   */
  const CuVector<BaseFloat>& GetCodeVecCorr() {
    return code_corr_;
  }

  void SetCodeVec(const CuVector<BaseFloat> &vec) {
    code_vec_.CopyFromVec(vec);
  }

  /*
   * Set the code vector to 0.
   */
  void ZeroCodeVec() {
    code_vec_.SetZero();
  }

  int32 GetCodeVecDim() {
    return code_dim_;
  }

  /*
   * Enable/Disable weight update.
   */
  void ConfigWeightUpdate(bool enable) {
    update_weight_ = enable;
  }

 protected:

  bool update_weight_;

  int32 code_dim_;  // dimensionality of the code

  CuMatrix<BaseFloat> augmented_in_;  // input feature + code
  CuVector<BaseFloat> code_vec_;  // code vector

  CuMatrix<BaseFloat> augmented_err_;  // the error includes code errors
  CuMatrix<BaseFloat> code_err_;  // error matrix for code vector

  CuVector<BaseFloat> code_corr_;  // correction of the code vector
};

}  // namespace

#endif /* NNET_CODEBL_H_ */
