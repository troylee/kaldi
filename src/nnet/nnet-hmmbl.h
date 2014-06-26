// nnet/nnet-hmmbl.h

/*
 * Created on: Sept. 12, 2013
 *     Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *     HMM based biased-linearity layer
 */
#ifndef KALDI_NNET_HMMBL_H
#define KALDI_NNET_HMMBL_H

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "vts/vts-first-order.h"

namespace kaldi {

class HMMBL : public UpdatableComponent {
 public:
  HMMBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : UpdatableComponent(dim_in, dim_out, nnet),
        linearity_(dim_out, dim_in),
        bias_(dim_out),
        linearity_corr_(dim_out, dim_in),
        bias_corr_(dim_out),
        linearity_cpu_(dim_out, dim_in),
        bias_cpu_(dim_out),
        apply_exp_(true)
  {
  }
  ~HMMBL()
  {
  }

  ComponentType GetType() const {
    return kHMMBL;
  }

  void ReadData(std::istream &is, bool binary) {

    // load in the HMM model

    trans_model_.Read(is, binary);
    am_gmm_clean_.Read(is, binary);

    KALDI_ASSERT(input_dim_ == 2 * am_gmm_clean_.Dim());
    KALDI_ASSERT(output_dim_ == am_gmm_clean_.NumGauss());

    ConvertWeight(am_gmm_clean_);

  }

  void WriteData(std::ostream &os, bool binary) const {
    trans_model_.Write(os, binary);
    am_gmm_clean_.Write(os, binary);

  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);

    if(apply_exp_){
      out->ApplyExp();
    }
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in_err,
                        CuMatrix<BaseFloat> *out_err) {
    // multiply error by weights
    out_err->AddMatMat(1.0, in_err, kNoTrans, linearity_, kNoTrans, 0.0);
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {

    KALDI_WARN<< "<hmmbl> layer is not updated.";
  }

  void VTSCompensate(const Vector<double> &mu_h, const Vector<double> &mu_z,
      const Vector<double> &var_z,
      int32 num_cepstral,
      int32 num_fbank,
      const Matrix<double> &dct_mat,
      const Matrix<double> &inv_dct_mat,
      std::vector<Matrix<double> > &Jx,
      std::vector<Matrix<double> > &Jz) {

    am_gmm_noisy_.CopyFromAmDiagGmm(am_gmm_clean_);
    CompensateModel(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat, inv_dct_mat,
        am_gmm_noisy_, Jx, Jz);

    ConvertWeight(am_gmm_noisy_);
  }

  void EnableExp(bool apply_exp){
    apply_exp_ = apply_exp;
  }

protected:

  /*
   * log N(x; mu, var) can be represented as w^T * [x^2 x 1]
   *
   * -0.5 * ( x^2/var - 2*mu/var + (mu^2/var + log(2*pi) + log(var))) )
   *
   */
  void ConvertWeight(const AmDiagGmm &am_gmm) {
    gmean_.Resize(am_gmm.Dim());
    gvar_.Resize(am_gmm.Dim());

    // initialize the weights, the -0.5 coefficient is ignored
    int32 num_pdf = am_gmm.NumPdfs();
    int32 dim = am_gmm.Dim();
    int32 hid = 0;
    for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
      int32 num_gauss = am_gmm.NumGaussInPdf(pdf);
      for (int32 ga = 0; ga < num_gauss; ++ga) {
        am_gmm.GetGaussianMean(pdf, ga, &gmean_);
        am_gmm.GetGaussianVariance(pdf, ga, &gvar_);

        gvar_.InvertElements();  // 1./ var

        (SubVector<BaseFloat>(linearity_cpu_.Row(hid), dim, dim)).CopyFromVec(
            gvar_);// coeff for x^2: 1./var

        (SubVector<BaseFloat>(linearity_cpu_.Row(hid), 0, dim)).CopyFromVec(
            gvar_);// coeff for x: 1./var
        (SubVector<BaseFloat>(linearity_cpu_.Row(hid), 0, dim)).MulElements(
            gmean_);// coeff for x: m./var
        (SubVector<BaseFloat>(linearity_cpu_.Row(hid), 0, dim)).Scale(-2);// coeff for x: -2*m./var

        gmean_.ApplyPow(2.0);
        gmean_.MulElements(gvar_);// m^2 ./ var
        bias_cpu_(hid) = gmean_.Sum() + dim * M_LOG_2PI - gvar_.SumLog();
        ++hid;
      }
    }

    linearity_cpu_.Scale(-0.5);
    bias_cpu_.Scale(-0.5);

    linearity_.CopyFromMat(linearity_cpu_);
    bias_.CopyFromVec(bias_cpu_);
  }

protected:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  AmDiagGmm am_gmm_clean_, am_gmm_noisy_;
  TransitionModel trans_model_;

  Matrix<BaseFloat> linearity_cpu_;
  Vector<BaseFloat> bias_cpu_;
  Vector<BaseFloat> gmean_, gvar_;

  bool apply_exp_; // only with exp, will the conversion equals to the likelihood

};

}  // namespace

#endif
