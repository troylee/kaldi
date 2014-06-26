// nnet/nnet-linbl.h

#ifndef KALDI_NNET_LINBL_H
#define KALDI_NNET_LINBL_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-biasedlinearity.h"

namespace kaldi {

class LinBL : public BiasedLinearity {
 public:
  LinBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : BiasedLinearity(dim_in, dim_out, nnet),
        lin_type_(0),
        num_blks_(0),
        blk_dim_(0),
        mask_(dim_out, dim_in),
        linearity_cpu_(dim_out, dim_in),
        bias_cpu_(dim_out)
  {
  }
  ~LinBL()
  {
  }

  ComponentType GetType() const {
    return kLinBL;
  }

  void SetLinBLType(int32 tid, int32 num_blks = 0, int32 blk_dim = 0) {
    KALDI_ASSERT(tid >= 0 && tid <= 3);
    lin_type_ = tid;

    if (tid == 1) { /* diagonal BL */
      Matrix<BaseFloat> mat(output_dim_, input_dim_, kSetZero);
      mat.SetUnit();
      mask_.CopyFromMat(mat);
    } else if (tid == 2 || tid == 3) {
      KALDI_ASSERT(
          num_blks >0 && blk_dim > 0 && num_blks * blk_dim == input_dim_);
      num_blks_ = num_blks;
      blk_dim_ = blk_dim;
      Matrix<BaseFloat> mat(output_dim_, input_dim_, kSetZero);
      for (int32 i = 0; i < num_blks_; ++i) {
        int32 offset = i * blk_dim_;
        for (int32 r = 0; r < blk_dim_; ++r) {
          for (int32 c = 0; c < blk_dim_; ++c) {
            mat(offset + r, offset + c) = 1.0;
          }
        }
      }
      mask_.CopyFromMat(mat);
    }
  }

  int32 GetLinBLType() const {
    return lin_type_;
  }

  void ReadData(std::istream &is, bool binary) {
    // read in LinBL type first
    ReadBasicType(is, binary, &lin_type_);
    if (lin_type_ == 2 || lin_type_ == 3) {
      ReadBasicType(is, binary, &num_blks_);
      ReadBasicType(is, binary, &blk_dim_);
    }

    BiasedLinearity::ReadData(is, binary);

    if (lin_type_ == 2 || lin_type_ == 3) {
      KALDI_ASSERT(num_blks_ * blk_dim_ == output_dim_);
    }

    // weight must be square matrix
    KALDI_ASSERT(linearity_.NumRows() == linearity_.NumCols());

    SetLinBLType(lin_type_, num_blks_, blk_dim_);

  }

  void WriteData(std::ostream &os, bool binary) const {

    WriteBasicType(os, binary, lin_type_);
    if (lin_type_ == 2 || lin_type_ == 3) {
      WriteBasicType(os, binary, num_blks_);
      WriteBasicType(os, binary, blk_dim_);
    }

    BiasedLinearity::WriteData(os, binary);
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {

    // compute gradient
    if (average_grad_) {
      linearity_corr_.AddMatMat(1.0 / input.NumRows(), err, kTrans, input,
                                kNoTrans,
                                momentum_);
      bias_corr_.AddRowSumMat(1.0 / input.NumRows(), err, momentum_);
    } else {
      linearity_corr_.AddMatMat(1.0, err, kTrans, input, kNoTrans, momentum_);
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

    // constrain the weights after each update
    switch (lin_type_) {
      case 0: /* standard BL, no constraints */
        break;
      case 1: /* diagonal BL */
      case 2: /* block diagonal BL */
        linearity_.MulElements(mask_);
        break;
      case 3: /* constrained block diagonal BL */
        linearity_.MulElements(mask_);
        /* average the blocks */
        linearity_.CopyToMat(&linearity_cpu_);
        bias_.CopyToVec(&bias_cpu_);
        SubMatrix<BaseFloat> blk0(linearity_cpu_, 0, blk_dim_, 0, blk_dim_);
        SubVector<BaseFloat> vec0(bias_cpu_, 0, blk_dim_);
        int32 offset;
        for (int32 i = 1; i < num_blks_; ++i) {
          offset = i * blk_dim_;
          blk0.AddMat(
              1.0,
              SubMatrix<BaseFloat>(linearity_cpu_, offset, blk_dim_, offset,
                                   blk_dim_));
          vec0.AddVec(1.0, SubVector<BaseFloat>(bias_cpu_, offset, blk_dim_));
        }
        blk0.Scale(1.0 / num_blks_);
        vec0.Scale(1.0 / num_blks_);
        /* copy back */
        for (int32 i = 1; i < num_blks_; ++i) {
          offset = i * blk_dim_;
          (SubMatrix<BaseFloat>(linearity_cpu_, offset, blk_dim_, offset,
                                blk_dim_)).CopyFromMat(blk0);
          (SubVector<BaseFloat>(bias_cpu_, offset, blk_dim_)).CopyFromVec(vec0);
        }
        linearity_.CopyFromMat(linearity_cpu_);
        bias_.CopyFromVec(bias_cpu_);
        break;
    }
  }

  /*
   * This function is used to tying the weights between different layers
   */
  void SetBiasWeight(const CuVector<BaseFloat> &bias) {
    bias_.CopyFromVec(bias);
  }

  void SetBiasWeight(const Vector<BaseFloat> &bias) {
    bias_.CopyFromVec(bias);
  }

  const CuVector<BaseFloat>& GetBiasWeight() {
    return bias_;
  }

 protected:
  /*
   * LinBL type code: 0 - standard BL;
   *                  1 - diagonal BL;
   *                  2 - block diagonal BL, requires num_blks_ and blk_dim_;
   *                  3 - shared block diagonal BL, requires num_blks_ and blk_dim_;
   */
  int32 lin_type_;

  int32 num_blks_;
  int32 blk_dim_;

  /* for constraining the weight */
  CuMatrix<BaseFloat> mask_;

  Matrix<BaseFloat> linearity_cpu_;
  Vector<BaseFloat> bias_cpu_;
};

}  // namespace

#endif
