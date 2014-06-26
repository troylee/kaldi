// nnet/nnet-linrbm.h

#ifndef KALDI_NNET_LINRBM_H
#define KALDI_NNET_LINRBM_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-rbm.h"

namespace kaldi {

class LinRbm : public Rbm {
 public:
  LinRbm(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : Rbm(dim_in, dim_out, nnet)
  {
    lin_type_ = 0;

    num_blks_ = 0;
    blk_dim_ = 0;

    /* initalize LIN */
    lin_linearity_cpu_.Resize(dim_in, dim_in, kSetZero);
    lin_linearity_cpu_.SetUnit();
    lin_linearity_.CopyFromMat(lin_linearity_cpu_);

    lin_bias_cpu_.Resize(dim_in, kSetZero);
    lin_bias_.CopyFromVec(lin_bias_cpu_);

    /* initialize mask, which will not be used for standard LIN */
    mask_.Resize(dim_in, dim_in);
  }

  ~LinRbm()
  {
  }

  ComponentType GetType() const {
    return kLinRbm;
  }

  void SetLinRbmType(int32 tid, int32 num_blks = 0, int32 blk_dim = 0) {
    KALDI_ASSERT(tid >= 0 && tid <= 3);
    lin_type_ = tid;

    if (tid == 1) { /* diagonal BL */
      Matrix<BaseFloat> mat(input_dim_, input_dim_, kSetZero);
      mat.SetUnit();
      mask_.CopyFromMat(mat);
    } else if (tid == 2 || tid == 3) {
      KALDI_ASSERT(
          num_blks >0 && blk_dim > 0 && num_blks * blk_dim == input_dim_);
      num_blks_ = num_blks;
      blk_dim_ = blk_dim;
      Matrix<BaseFloat> mat(input_dim_, input_dim_, kSetZero);
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

  int32 GetLinRbmType() {
    return lin_type_;
  }

  int32 GetLinRbmNumBlks() {
    return num_blks_;
  }

  int32 GetLinRbmBlkDim() {
    return blk_dim_;
  }

  // initialize LinRbm with a plain Rbm
  void ReadRbm(std::istream &is, bool binary) {

    std::string token;

    int first_char = Peek(is, binary);
    if (first_char == EOF) {
      KALDI_ERR<< "Empty model file!";
    }

    ReadToken(is, binary, &token);
    Component::ComponentType comp_type = Component::MarkerToType(token);

    KALDI_ASSERT(comp_type == Component::kRbm);

    ReadBasicType(is, binary, &output_dim_);
    ReadBasicType(is, binary, &input_dim_);

    Rbm::ReadData(is, binary);

    /* intialize LIN type */
    lin_type_=0;
    num_blks_=0;
    blk_dim_=0;

    /* initalize LIN */
    lin_linearity_cpu_.Resize(input_dim_, input_dim_, kSetZero);
    lin_linearity_cpu_.SetUnit();
    lin_linearity_.CopyFromMat(lin_linearity_cpu_);

    lin_bias_cpu_.Resize(input_dim_, kSetZero);
    lin_bias_.CopyFromVec(lin_bias_cpu_);


  }

  void ReadData(std::istream &is, bool binary) {
    // read in LinBL type first
    ReadBasicType(is, binary, &lin_type_);
    if (lin_type_ == 2 || lin_type_ == 3) {
      ReadBasicType(is, binary, &num_blks_);
      ReadBasicType(is, binary, &blk_dim_);
    }

    Rbm::ReadData(is, binary);

    if (lin_type_ == 2 || lin_type_ == 3) {
      KALDI_ASSERT(num_blks_ * blk_dim_ == input_dim_);
    }

    lin_linearity_.Read(is, binary);
    lin_bias_.Read(is, binary);

    KALDI_ASSERT(
        lin_linearity_.NumRows() == input_dim_ && lin_linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(lin_bias_.Dim() == input_dim_);

    SetLinRbmType(lin_type_, num_blks_, blk_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    // write LinBL type first
    WriteBasicType(os, binary, lin_type_);
    if (lin_type_ == 2 || lin_type_ == 3) {
      WriteBasicType(os, binary, num_blks_);
      WriteBasicType(os, binary, blk_dim_);
    }

    Rbm::WriteData(os, binary);

    lin_linearity_.Write(os, binary);
    lin_bias_.Write(os, binary);

  }

  void SetLinLinearityWeight(const Matrix<BaseFloat> &weight, bool trans) {
    if (trans) {
      Matrix<BaseFloat> mat;
      mat.CopyFromMat(weight, kTrans);
      lin_linearity_.CopyFromMat(mat);
    } else {
      lin_linearity_.CopyFromMat(weight);
    }
  }

  void SetLinLinearityWeight(const CuMatrix<BaseFloat> &weight, bool trans) {
    if (trans) {
      Matrix<BaseFloat> mat;
      weight.CopyToMat(&mat);
      mat.Transpose();
      lin_linearity_.CopyFromMat(mat);
    } else {
      lin_linearity_.CopyFromMat(weight);
    }
  }

  const CuMatrix<BaseFloat>& GetLinLinearityWeight() {
    return lin_linearity_;
  }

  void SetLinBiasWeight(const Vector<BaseFloat> &bias) {
    lin_bias_.CopyFromVec(bias);
  }

  void SetLinBiasWeight(const CuVector<BaseFloat> &bias) {
    lin_bias_.CopyFromVec(bias);
  }

  const CuVector<BaseFloat>& GetLinBiasWeight() {
    return lin_bias_;
  }

  void SetToIdentity(){
    Matrix<BaseFloat> mat(lin_linearity_.NumRows(), lin_linearity_.NumCols());
    mat.SetUnit();
    lin_linearity_.CopyFromMat(mat);
    lin_bias_.SetZero();
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    if (tmp_in_.NumRows() != in.NumRows()
        || tmp_in_.NumCols() != in.NumCols()) {
      tmp_in_.Resize(in.NumRows(), in.NumCols());
    }
    // LIN transform
    tmp_in_.AddVecToRows(1.0, lin_bias_, 0.0);
    tmp_in_.AddMatMat(1.0, in, kNoTrans, lin_linearity_, kTrans, 1.0);

    // precopy bias
    out->AddVecToRows(1.0, hid_bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, tmp_in_, kNoTrans, vis_hid_, kTrans, 1.0);
    // optionally apply sigmoid
    if (hid_type_ == RbmBase::BERNOULLI) {
      cu::Sigmoid(*out, out);
    }
  }

  // RBM training API
  void Reconstruct(const CuMatrix<BaseFloat> &hid_state,
                   CuMatrix<BaseFloat> *vis_probs) {
    // check the dim
    if (output_dim_ != hid_state.NumCols()) {
      KALDI_ERR<< "Nonmatching dims, component:" << output_dim_ << " data:" << hid_state.NumCols();
    }
    // optionally allocate buffer
    if (input_dim_ != vis_probs->NumCols() || hid_state.NumRows() != vis_probs->NumRows()) {
      vis_probs->Resize(hid_state.NumRows(), input_dim_);
    }
    if (input_dim_ != tmp_in_.NumCols() || hid_state.NumRows() != tmp_in_.NumRows()) {
      tmp_in_.Resize(hid_state.NumRows(), input_dim_);
    }

    // precopy bias
    tmp_in_.AddVecToRows(1.0, vis_bias_, 0.0);
    // multiply by weights
    tmp_in_.AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // optionally apply sigmoid
    if (vis_type_ == RbmBase::BERNOULLI) {
      cu::Sigmoid(tmp_in_, &tmp_in_);
    }

    // LIN
    vis_probs->AddVecToRows(1.0, lin_bias_, 0.0);
    vis_probs->AddMatMat(1.0, tmp_in_, kNoTrans, lin_linearity_, kNoTrans, 1.0);

  }

  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {

    assert(pos_vis.NumRows() == pos_hid.NumRows() &&
        pos_vis.NumRows() == neg_vis.NumRows() &&
        pos_vis.NumRows() == neg_hid.NumRows() &&
        pos_vis.NumCols() == neg_vis.NumCols() &&
        pos_hid.NumCols() == neg_hid.NumCols() &&
        pos_vis.NumCols() == input_dim_ &&
        pos_hid.NumCols() == output_dim_);

    //lazy initialization of buffers (possibly reduces to no-op)
    lin_linearity_corr_.Resize(lin_linearity_.NumRows(), lin_linearity_.NumCols());
    lin_bias_corr_.Resize(lin_bias_.Dim());
    tmp_in_.Resize(pos_vis.NumRows(), input_dim_);

    //  UPDATE vishid matrix
    //
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    //
    //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
    //                 +(epsilonw/numcases)*posprods
    //                 -(epsilonw*weightcost)*vishid[t-1]
    //
    BaseFloat N = static_cast<BaseFloat>(pos_vis.NumRows());
    // pass through the original RBM
    tmp_in_.AddVecToRows(1.0, vis_bias_, 0.0);
    tmp_in_.AddMatMat(1.0, neg_hid, kNoTrans, vis_hid_, kNoTrans, 1.0);
    lin_linearity_corr_.AddMatMat(-learn_rate_/N, tmp_in_, kTrans, neg_vis, kNoTrans, momentum_);
    lin_bias_corr_.AddRowSumMat(-learn_rate_/N, tmp_in_, momentum_);

    tmp_in_.AddVecToRows(1.0, vis_bias_, 0.0);
    tmp_in_.AddMatMat(1.0, pos_hid, kNoTrans, vis_hid_, kNoTrans, 1.0);
    lin_linearity_corr_.AddMatMat(+learn_rate_/N, tmp_in_, kTrans, pos_vis, kNoTrans, 1.0);
    lin_bias_corr_.AddRowSumMat(+learn_rate_/N, tmp_in_, 1.0);

    lin_linearity_corr_.AddMat(-learn_rate_*l2_penalty_, lin_linearity_, 1.0);
    lin_linearity_.AddMat(1.0, lin_linearity_corr_, 1.0);

    lin_bias_.AddVec(1.0, lin_bias_corr_, 1.0);

    // constrain the weights after each update
    switch (lin_type_) {
      case 0: /* standard BL, no constraints */
      break;
      case 1: /* diagonal BL */
      case 2: {
        /* block diagonal BL */
        lin_linearity_.MulElements(mask_);
      }
      break;
      case 3: {
        /* constrained block diagonal BL */
        lin_linearity_.MulElements(mask_);
        /* average the blocks */
        lin_linearity_.CopyToMat(&lin_linearity_cpu_);
        lin_bias_.CopyToVec(&lin_bias_cpu_);
        SubMatrix<BaseFloat> blk0(lin_linearity_cpu_, 0, blk_dim_, 0, blk_dim_);
        SubVector<BaseFloat> vec0(lin_bias_cpu_, 0, blk_dim_);
        int32 offset;
        for(int32 i=1; i<num_blks_; ++i) {
          offset = i * blk_dim_;
          blk0.AddMat(1.0, SubMatrix<BaseFloat>(lin_linearity_cpu_, offset, blk_dim_, offset, blk_dim_));
          vec0.AddVec(1.0, SubVector<BaseFloat>(lin_bias_cpu_, offset, blk_dim_));
        }
        blk0.Scale(1.0/num_blks_);
        vec0.Scale(1.0/num_blks_);
        /* copy back */
        for(int32 i=1; i<num_blks_; ++i) {
          offset= i* blk_dim_;
          (SubMatrix<BaseFloat>(lin_linearity_cpu_, offset, blk_dim_, offset, blk_dim_)).CopyFromMat(blk0);
          (SubVector<BaseFloat>(lin_bias_cpu_, offset, blk_dim_)).CopyFromVec(vec0);
        }
		lin_linearity_.CopyFromMat(lin_linearity_cpu_);
		lin_bias_.CopyFromVec(lin_bias_cpu_);
      }
      break;
    }
  }

private:

  /*
   * LinBL type code: 0 - standard LIN;
   *                  1 - diagonal LIN;
   *                  2 - block diagonal LIN, requires num_blks_ and blk_dim_;
   *                  3 - shared block diagonal LIN, requires num_blks_ and blk_dim_;
   */
  int32 lin_type_;

  int32 num_blks_;
  int32 blk_dim_;

  /* for constraining the weight */
  CuMatrix<BaseFloat> mask_;

  /* LIN weights */
  CuMatrix<BaseFloat> lin_linearity_;
  CuVector<BaseFloat> lin_bias_;

  /* weight updates */
  CuMatrix<BaseFloat> lin_linearity_corr_;
  CuVector<BaseFloat> lin_bias_corr_;

  /* matrix for LIN transformed input */
  CuMatrix<BaseFloat> tmp_in_;

  Matrix<BaseFloat> lin_linearity_cpu_;
  Vector<BaseFloat> lin_bias_cpu_;

};

}  // namespace

#endif
