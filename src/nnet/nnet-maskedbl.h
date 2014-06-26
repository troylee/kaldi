// nnet/nnet-maskedbl.h

#ifndef KALDI_NNET_MASKEDBL_H
#define KALDI_NNET_MASKEDBL_H

#include "nnet/nnet-biasedlinearity.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class MaskedBL : public BiasedLinearity {
 public:
  MaskedBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : BiasedLinearity(dim_in, dim_out, nnet),
        mask_(dim_out, dim_in),
        cpu_linearity_(dim_out, dim_in),
        kernel_sharing_(false),
        num_kernels_(0),
        kernel_rows_(0),
        kernel_cols_(0)
  {
  }
  ~MaskedBL()
  {
  }

  ComponentType GetType() const {
    return kMaskedBL;
  }

  void ReadData(std::istream &is, bool binary) {

    ReadBasicType(is, binary, &kernel_sharing_);
    ReadBasicType(is, binary, &num_kernels_);
    ReadBasicType(is, binary, &kernel_rows_);
    ReadBasicType(is, binary, &kernel_cols_);

    BiasedLinearity::ReadData(is, binary);

    mask_.Read(is, binary);

    KALDI_ASSERT(mask_.NumRows() == output_dim_);
    KALDI_ASSERT(mask_.NumCols() == input_dim_);
    KALDI_ASSERT(
        num_kernels_ == 0 || (kernel_rows_ * num_kernels_ == output_dim_ && kernel_cols_*num_kernels_ == input_dim_));
    if (num_kernels_ > 0) {
      kernel_.Resize(kernel_rows_, kernel_cols_, kSetZero);
    }
  }

  void WriteData(std::ostream &os, bool binary) const {

    WriteBasicType(os, binary, kernel_sharing_);
    WriteBasicType(os, binary, num_kernels_);
    WriteBasicType(os, binary, kernel_rows_);
    WriteBasicType(os, binary, kernel_cols_);

    BiasedLinearity::WriteData(os, binary);

    mask_.Write(os, binary);

  }

  /*
   * Only the updat is different from the biased linearity layer
   */
  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {

    BiasedLinearity::Update(input, err);

    // apply mask to the new weight
    linearity_.MulElements(mask_);

    // apply diagonal block sharing
    if (kernel_sharing_) {
      ApplySharing();
    }
  }

  void SetMask(const Matrix<BaseFloat> &mask) {
    KALDI_ASSERT(mask.NumRows() == output_dim_);
    KALDI_ASSERT(mask.NumCols() == input_dim_);

    mask_.CopyFromMat(mask);
  }

  void SetSharing(int32 num_kernels, int32 kernel_rows, int32 kernel_cols) {
    KALDI_ASSERT(
        num_kernels>0 && kernel_rows * num_kernels == output_dim_ && kernel_cols*num_kernels == input_dim_);
    num_kernels_ = num_kernels;
    kernel_rows_ = kernel_rows;
    kernel_cols_ = kernel_cols;
    kernel_sharing_ = true;
    kernel_.Resize(kernel_rows_, kernel_cols_);
  }

  void ApplySharing() {
    linearity_.CopyToMat(&cpu_linearity_);
    kernel_.SetZero();
    for (int32 n = 0; n < num_kernels_; ++n) {
      kernel_.AddMat(
          1.0,
          SubMatrix<BaseFloat>(cpu_linearity_, n * kernel_rows_, kernel_rows_,
                               n * kernel_cols_, kernel_cols_));
    }
    kernel_.Scale(1.0 / num_kernels_);

    for (int32 n = 0; n < num_kernels_; ++n) {
      (SubMatrix<BaseFloat>(cpu_linearity_, n * kernel_rows_, kernel_rows_,
                            n * kernel_cols_, kernel_cols_)).CopyFromMat(
          kernel_, kNoTrans);
    }
    linearity_.CopyFromMat(cpu_linearity_);

  }

 private:
  CuMatrix<BaseFloat> mask_;

  Matrix<BaseFloat> cpu_linearity_;
  Matrix<BaseFloat> kernel_;

  bool kernel_sharing_;
  int32 num_kernels_;
  int32 kernel_rows_;
  int32 kernel_cols_;

};

}  // namespace

#endif
