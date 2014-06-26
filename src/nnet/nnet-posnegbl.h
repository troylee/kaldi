// nnet/nnet-posnegbl.h

#ifndef KALDI_NNET_POSNEGBL_H
#define KALDI_NNET_POSNEGBL_H

#include "nnet/nnet-component.h"
#include "gmm/am-diag-gmm.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class PosNegBL : public UpdatableComponent {

 public:
  PosNegBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : UpdatableComponent(dim_in, dim_out, nnet),
        cpu_linearity_(dim_out, dim_in),
        cpu_bias_(dim_out),
        linearity_(dim_out, dim_in),
        bias_(dim_out),
        cpu_linearity_corr_(dim_out, dim_in),
        cpu_bias_corr_(dim_out),
        linearity_corr_(dim_out, dim_in),
        bias_corr_(dim_out),
        pos2neg_log_prior_ratio_(dim_out),
        var_scale_(dim_out),
        num_cepstral_(-1),
        num_fbank_(-1),
        ceplifter_(-1),
        compensate_var_(true),
        pos_var_weight_(1.0)
  {
  }
  ~PosNegBL()
  {
  }

  ComponentType GetType() const {
    return kPosNegBL;
  }

  void CreateModel(int32 num_frame, int32 delta_order, int32 num_cepstral,
                   int32 num_fbank,
                   BaseFloat ceplifter,
                   const Vector<double> &pos2neg_log_prior_ratio,
                   const Vector<double> &var_scale,
                   const AmDiagGmm &pos_am,
                   const AmDiagGmm &neg_am) {
    num_frame_ = num_frame;
    delta_order_ = delta_order;
    num_cepstral_ = num_cepstral;
    num_fbank_ = num_fbank;
    ceplifter_ = ceplifter;
    pos2neg_log_prior_ratio_.CopyFromVec(pos2neg_log_prior_ratio);
    var_scale_.CopyFromVec(var_scale);
    pos_am_gmm_.CopyFromAmDiagGmm(pos_am);
    neg_am_gmm_.CopyFromAmDiagGmm(neg_am);

    KALDI_ASSERT(pos2neg_log_prior_ratio_.Dim() == output_dim_);
    KALDI_ASSERT(var_scale_.Dim() == output_dim_);
    KALDI_ASSERT(
        pos_am_gmm_.NumPdfs() == output_dim_ && pos_am_gmm_.Dim() == input_dim_);
    KALDI_ASSERT(
        neg_am_gmm_.NumPdfs() == output_dim_ && neg_am_gmm_.Dim() == input_dim_);

    PrepareDCTXforms();
    ConvertPosNegGaussianToNNLayer(pos_am_gmm_, neg_am_gmm_,
                                   pos2neg_log_prior_ratio_,
                                   var_scale_,
                                   cpu_linearity_,
                                   cpu_bias_);
    linearity_.CopyFromMat(cpu_linearity_);
    bias_.CopyFromVec(cpu_bias_);
  }

  void ReadData(std::istream &is, bool binary) {
    ReadBasicType(is, binary, &num_frame_);
    ReadBasicType(is, binary, &delta_order_);
    ReadBasicType(is, binary, &num_cepstral_);
    ReadBasicType(is, binary, &num_fbank_);
    ReadBasicType(is, binary, &ceplifter_);

    pos2neg_log_prior_ratio_.Read(is, binary);
    var_scale_.Read(is, binary);
    pos_am_gmm_.Read(is, binary);
    neg_am_gmm_.Read(is, binary);

    KALDI_ASSERT(pos2neg_log_prior_ratio_.Dim() == output_dim_);
    KALDI_ASSERT(var_scale_.Dim() == output_dim_);
    KALDI_ASSERT(
        pos_am_gmm_.NumPdfs() == output_dim_ && pos_am_gmm_.Dim() == input_dim_);
    KALDI_ASSERT(
        neg_am_gmm_.NumPdfs() == output_dim_ && neg_am_gmm_.Dim() == input_dim_);

    PrepareDCTXforms();
    ConvertPosNegGaussianToNNLayer(pos_am_gmm_, neg_am_gmm_,
                                   pos2neg_log_prior_ratio_,
                                   var_scale_,
                                   cpu_linearity_,
                                   cpu_bias_);
    linearity_.CopyFromMat(cpu_linearity_);
    bias_.CopyFromVec(cpu_bias_);

  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, num_frame_);
    WriteBasicType(os, binary, delta_order_);
    WriteBasicType(os, binary, num_cepstral_);
    WriteBasicType(os, binary, num_fbank_);
    WriteBasicType(os, binary, ceplifter_);
    pos2neg_log_prior_ratio_.Write(os, binary);
    var_scale_.Write(os, binary);
    pos_am_gmm_.Write(os, binary);
    neg_am_gmm_.Write(os, binary);
  }

  // CPU based forward
  void Forward(const Matrix<BaseFloat> &in, Matrix<BaseFloat> *out) {
    // precopy bias
    out->SetZero();
    out->AddVecToRows(1.0, cpu_bias_);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, cpu_linearity_, kTrans, 1.0);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // multiply error by weights
    out_err->AddMatMat(1.0, in_err, kNoTrans, linearity_, kNoTrans, 0.0);
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

    linearity_corr_.CopyToMat(&cpu_linearity_corr_);
    bias_corr_.CopyToVec(&cpu_bias_corr_);

    // update the log ratio, the gradient equals to bias gradient
    pos2neg_log_prior_ratio_.AddVec(-learn_rate_, cpu_bias_corr_);

    // update var scale
    //UpdateVarScale();

    /*
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
     */
  }

  void SetUpdateFlag(std::string flag) {

  }

  void PrepareDCTXforms();

  void SetNoise(bool compensate_var, const Vector<double> &mu_h,
                const Vector<double> &mu_z,
                const Vector<double> &var_z,
                BaseFloat pos_var_weight = 1.0);

  void GetNoise(Vector<double> &mu_h, Vector<double> &mu_z, Vector<double> &var_z) {
    mu_h.CopyFromVec(mu_h_);
    mu_z.CopyFromVec(mu_z_);
    var_z.CopyFromVec(var_z_);
  }

 private:

  void UpdateVarScale();

  void CompensateMultiFrameGmm(const Vector<double> &mu_h,
                               const Vector<double> &mu_z,
                               const Vector<double> &var_z, bool compensate_var,
                               int32 num_cepstral,
                               int32 num_fbank,
                               const Matrix<double> &dct_mat,
                               const Matrix<double> &inv_dct_mat,
                               int32 num_frames,
                               AmDiagGmm &noise_am_gmm);

  void InterpolateVariance(BaseFloat pos_weight, AmDiagGmm &pos_am_gmm,
                           AmDiagGmm &neg_am_gmm);

  void ConvertPosNegGaussianToNNLayer(
      const AmDiagGmm &pos_am_gmm,
      const AmDiagGmm &neg_am_gmm,
      const Vector<double> &pos2neg_log_prior_ratio,
      const Vector<double> &var_scale,
      Matrix<BaseFloat> &linearity,
      Vector<BaseFloat> &bias);

 private:
  Matrix<BaseFloat> cpu_linearity_;
  Vector<BaseFloat> cpu_bias_;

  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  Matrix<BaseFloat> cpu_linearity_corr_;
  Vector<BaseFloat> cpu_bias_corr_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  Vector<double> pos2neg_log_prior_ratio_;
  Vector<double> var_scale_;

  AmDiagGmm pos_am_gmm_, pos_noise_am_;
  AmDiagGmm neg_am_gmm_, neg_noise_am_;

  // parameters for VTS compensation
  int32 num_frame_;  // multiple frame input
  int32 delta_order_;  // how many deltas, currently only supports 2, i.e. static(0), delta(1) and acc(2)
  int32 num_cepstral_;
  int32 num_fbank_;
  BaseFloat ceplifter_;
  Matrix<double> dct_mat_;
  Matrix<double> inv_dct_mat_;

  // noise parameters
  bool compensate_var_;
  BaseFloat pos_var_weight_;
  Vector<double> mu_h_;
  Vector<double> mu_z_;
  Vector<double> var_z_;
};

}  // namespace

#endif
