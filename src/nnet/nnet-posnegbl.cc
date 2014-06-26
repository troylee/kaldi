// nnet/nnet-posnegbl.cc

#include "nnet/nnet-posnegbl.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/diag-gmm.h"

namespace kaldi {

void PosNegBL::UpdateVarScale() {
  Vector<double> corr(var_scale_.Dim(), kSetZero);
  Vector<double> mu(input_dim_, kSetZero);
  Vector<double> grad(input_dim_, kSetZero);

  int32 num_pdf = pos_noise_am_.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {

    // iterate all the Gaussians
    const DiagGmm *gmm_pos = &(pos_noise_am_.GetPdf(pdf));
    DiagGmmNormal ngmm_pos(*gmm_pos);

    const DiagGmm *gmm_neg = &(neg_noise_am_.GetPdf(pdf));
    DiagGmmNormal ngmm_neg(*gmm_neg);

    KALDI_ASSERT(gmm_pos->NumGauss()==1 && gmm_neg->NumGauss()==1);

    mu.CopyRowFromMat(ngmm_pos.means_, 0);
    mu.AddVec(-1.0, ngmm_neg.means_.Row(0));
    mu.DivElements(ngmm_pos.vars_.Row(0));
    mu.MulElements(cpu_linearity_corr_.Row(pdf));

    corr(pdf) = mu.Sum();

    mu.SetZero();
    mu.AddVec2(1.0, ngmm_pos.means_.Row(0));
    mu.AddVec2(-1.0, ngmm_neg.means_.Row(0));
    mu.DivElements(ngmm_pos.vars_.Row(0));

    corr(pdf) -= 0.5 * mu.Sum();
  }

  var_scale_.AddVec(-learn_rate_, corr);

}

void PosNegBL::PrepareDCTXforms() {

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG<< "Generate DCT matrics ...";
  }

  // compute the DCT matrix, in HTK C0 is the last component, i.e. C1, C2, ..., C12, C0
  if (dct_mat_.NumRows() != num_cepstral_ || dct_mat_.NumCols() != num_fbank_) {
    dct_mat_.Resize(num_cepstral_, num_fbank_, kSetZero);
  } else {
    dct_mat_.SetZero();
  }

  if (inv_dct_mat_.NumRows() != num_fbank_
      || inv_dct_mat_.NumCols() != num_cepstral_) {
    inv_dct_mat_.Resize(num_fbank_, num_cepstral_, kSetZero);
  } else {
    inv_dct_mat_.SetZero();
  }

  Vector<double> ceplifter_vec(num_cepstral_);
  double normalizer = sqrt(2.0 / static_cast<double>(num_fbank_));  // normalizer for elements.
  // cepstral liftering is not applied to the C0
  ceplifter_vec(num_cepstral_ - 1) = 1.0;
  // In HTK C0 is using sqrt(2/N), while in others are all sqrt(1/N)
  // As the features we are using is from HTK, it is thus better to use the same value.
  // although in stardard DCT, for the 0th elements, the constant should be
  // sqrt(1.0 / static_cast<double>(num_fbank))
  for (int32 j = 0; j < num_fbank_; ++j) {
    dct_mat_(num_cepstral_ - 1, j) = normalizer;
  }

  for (int32 k = 1; k < num_cepstral_; ++k) {
    ceplifter_vec(k - 1) = 1.0
    + 0.5 * ceplifter_ * sin(static_cast<double>(M_PI) * k / ceplifter_);
    for (int32 n = 0; n < num_fbank_; ++n)
    dct_mat_(k - 1, n) = normalizer
    * cos(static_cast<double>(M_PI) / num_fbank_ * (n + 0.5) * k);
  }

  if (g_kaldi_verbose_level >= 2) {
    KALDI_LOG << "DCT Transform: " << dct_mat_;
    KALDI_LOG << "Cepstal Lifter Vector: " << ceplifter_vec;
  }

  // generate the inverse transforms, pinv(C)=C_T * inv(C * C_T)
  Matrix<double> c_ct(num_cepstral_, num_cepstral_);
  c_ct.AddMatMat(1.0, dct_mat_, kNoTrans, dct_mat_, kTrans, 0.0);
  c_ct.Invert();
  inv_dct_mat_.AddMatMat(1.0, dct_mat_, kTrans, c_ct, kNoTrans, 0.0);

  if (g_kaldi_verbose_level >= 2) {
    KALDI_LOG << "Inverse DCT: " << inv_dct_mat_;
  }

  // multiply the ceplifter coefficients to the DCT and inverse DCT matrix for simplicity.
  // i.e. set C=L*C, inv_C = inv_C * inv_L
  // as L is diagonal matrix, the multiplication is simply the scaling of each row
  if (ceplifter_ > 0) {
    dct_mat_.MulRowsVec(ceplifter_vec);
    ceplifter_vec.InvertElements();
    inv_dct_mat_.MulColsVec(ceplifter_vec);

    if (g_kaldi_verbose_level >= 2) {
      KALDI_LOG << "DCT Transform with CepLifter: " << dct_mat_;
      KALDI_LOG << "Inverse DCT Transform with CepLifter: " << inv_dct_mat_;
    }
  }

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG << "DCT matrices generation done!";
  }
}

void PosNegBL::SetNoise(bool compensate_var, const Vector<double> &mu_h,
                        const Vector<double> &mu_z,
                        const Vector<double> &var_z,
                        BaseFloat pos_var_weight) {


  if (num_cepstral_ <= 0 || num_fbank_ <= 0) {
    KALDI_ERR<< "DCT Transforms are not prepared yet!";
  }
  compensate_var_ = compensate_var;
  pos_var_weight_ = pos_var_weight;
  if(mu_h_.Dim()!=mu_h.Dim()) {
    mu_h_.Resize(mu_h.Dim());
  }
  if (mu_z_.Dim()!=mu_z.Dim()) {
    mu_z_.Resize(mu_z.Dim());
  }
  if(var_z_.Dim()!=var_z.Dim()) {
    var_z_.Resize(var_z.Dim());
  }
  mu_h_.CopyFromVec(mu_h);
  mu_z_.CopyFromVec(mu_z);
  var_z_.CopyFromVec(var_z);

  pos_noise_am_.CopyFromAmDiagGmm(pos_am_gmm_);
  neg_noise_am_.CopyFromAmDiagGmm(neg_am_gmm_);

  CompensateMultiFrameGmm(mu_h_, mu_z_, var_z_, compensate_var_, num_cepstral_, num_fbank_, dct_mat_, inv_dct_mat_, num_frame_, pos_noise_am_);
  CompensateMultiFrameGmm(mu_h_, mu_z_, var_z_, compensate_var_, num_cepstral_, num_fbank_, dct_mat_, inv_dct_mat_, num_frame_, neg_noise_am_);

  InterpolateVariance(pos_var_weight_, pos_noise_am_, neg_noise_am_);

  ConvertPosNegGaussianToNNLayer(pos_noise_am_, neg_noise_am_, pos2neg_log_prior_ratio_, var_scale_, cpu_linearity_, cpu_bias_);

  //KALDI_LOG << "PosNegBL Compensated weight: " << cpu_linearity_;
  //KALDI_LOG << "PosNegBL Compensated bias: " << cpu_bias_;

  linearity_.CopyFromMat(cpu_linearity_);
  bias_.CopyFromVec(cpu_bias_);
}

void PosNegBL::CompensateMultiFrameGmm(const Vector<double> &mu_h,
                                       const Vector<double> &mu_z,
                                       const Vector<double> &var_z,
                                       bool compensate_var,
                                       int32 num_cepstral,
                                       int32 num_fbank,
                                       const Matrix<double> &dct_mat,
                                       const Matrix<double> &inv_dct_mat,
                                       int32 num_frames,
                                       AmDiagGmm &noise_am_gmm) {

  //KALDI_LOG << "Beginning compensate model ...";
  Matrix<double> Jx(num_cepstral, num_cepstral, kSetZero);
  Matrix<double> Jz(num_cepstral, num_cepstral, kSetZero);

  int32 feat_dim = noise_am_gmm.Dim() / num_frames;
  KALDI_ASSERT(feat_dim * num_frames == noise_am_gmm.Dim());

  // iterate all the GMMs
  int32 num_pdf = noise_am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *gmm = &(noise_am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm(*gmm);

    int32 num_gauss = gmm->NumGauss();
    for (int32 g = 0; g < num_gauss; ++g) {

      // compute the necessary transforms
      Vector<double> mu_y_s(num_cepstral);
      Vector<double> tmp_fbank(num_fbank);

      for (int32 n = 0; n < num_frames; ++n) {
        SubVector<double> cur_mean(ngmm.means_.Row(g), n * feat_dim, feat_dim);
        SubVector<double> cur_var(ngmm.vars_.Row(g), n * feat_dim, feat_dim);

        for (int32 ii = 0; ii < num_cepstral; ++ii) {
          mu_y_s(ii) = mu_z(ii) - cur_mean(ii) - mu_h(ii);
        }  // mu_n - mu_x - mu_h
        tmp_fbank.AddMatVec(1.0, inv_dct_mat, kNoTrans, mu_y_s, 0.0);  // C_inv * (mu_n - mu_x - mu_h)
        tmp_fbank.ApplyExp();  // exp( C_inv * (mu_n - mu_x - mu_h) )
        tmp_fbank.Add(1.0);  // 1 + exp( C_inv * (mu_n - mu_x - mu_h) )
        Vector<double> tmp_inv(tmp_fbank);  // keep a version
        tmp_fbank.ApplyLog();  // log ( 1 + exp( C_inv * (mu_n - mu_x - mu_h) ) )
        tmp_inv.InvertElements();  // 1.0 / ( 1 + exp( C_inv * (mu_n - mu_x - mu_h) ) )

        // new static mean
        for (int32 ii = 0; ii < num_cepstral; ++ii) {
          mu_y_s(ii) = cur_mean(ii) + mu_h(ii);
        }  // mu_x + mu_h
        mu_y_s.AddMatVec(1.0, dct_mat, kNoTrans, tmp_fbank, 1.0);  // mu_x + mu_h + C * log ( 1 + exp( C_inv * (mu_n - mu_x - mu_h) ) )

        // compute J
        Matrix<double> tmp_dct(dct_mat);
        tmp_dct.MulColsVec(tmp_inv);
        Jx.AddMatMat(1.0, tmp_dct, kNoTrans, inv_dct_mat, kNoTrans, 0.0);

        // compute I_J
        Jz.CopyFromMat(Jx);
        for (int32 ii = 0; ii < num_cepstral; ++ii)
          Jz(ii, ii) = 1.0 - Jz(ii, ii);

        // compute and update mean
        if (g_kaldi_verbose_level >= 9) {
          KALDI_LOG<< "Mean Before: " << cur_mean;
        }
        Vector<double> tmp_mu(num_cepstral);
        SubVector<double> mu_s(cur_mean, 0, num_cepstral);
        mu_s.CopyFromVec(mu_y_s);
        SubVector<double> mu_dt(cur_mean, num_cepstral, num_cepstral);
        tmp_mu.CopyFromVec(mu_dt);
        mu_dt.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
        SubVector<double> mu_acc(cur_mean, 2 * num_cepstral, num_cepstral);
        tmp_mu.CopyFromVec(mu_acc);
        mu_acc.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
        if (g_kaldi_verbose_level >= 9) {
          KALDI_LOG<< "Mean After: " << cur_mean;
        }

        if (compensate_var) {
          // compute and update covariance
          if (g_kaldi_verbose_level >= 9) {
            KALDI_LOG<< "Covarianc Before: " << cur_var;
          }
          for (int32 ii = 0; ii < 3; ++ii) {
            Matrix<double> tmp_var1(Jx), tmp_var2(Jz), new_var(num_cepstral,
                num_cepstral);
            SubVector<double> x_var(cur_var, ii * num_cepstral, num_cepstral);
            SubVector<double> n_var(var_z, ii * num_cepstral, num_cepstral);

            tmp_var1.MulColsVec(x_var);
            new_var.AddMatMat(1.0, tmp_var1, kNoTrans, Jx, kTrans, 0.0);

            tmp_var2.MulColsVec(n_var);
            new_var.AddMatMat(1.0, tmp_var2, kNoTrans, Jz, kTrans, 1.0);

            x_var.CopyDiagFromMat(new_var);
          }

          if (g_kaldi_verbose_level >= 9) {
            KALDI_LOG << "Covariance After: " << cur_var;
          }
        }

      }
    }

    ngmm.CopyToDiagGmm(gmm);
    gmm->ComputeGconsts();
  }

  //KALDI_LOG << "Model compensation done!";
}

void PosNegBL::InterpolateVariance(BaseFloat pos_weight, AmDiagGmm &pos_am_gmm,
                                   AmDiagGmm &neg_am_gmm) {

  KALDI_ASSERT(pos_weight >= 0.0 && pos_weight<=1.0);

  // iterate all the GMMs
  int32 num_pdf = pos_am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal pos_ngmm(*pos_gmm);
    KALDI_ASSERT(pos_gmm->NumGauss()==1);

    DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal neg_ngmm(*neg_gmm);
    KALDI_ASSERT(neg_gmm->NumGauss()==1);

    (pos_ngmm.vars_.Row(0)).Scale(pos_weight);
    (pos_ngmm.vars_.Row(0)).AddVec(1 - pos_weight, neg_ngmm.vars_.Row(0));

    neg_ngmm.vars_.CopyRowFromVec(pos_ngmm.vars_.Row(0), 0);

    pos_ngmm.CopyToDiagGmm(pos_gmm);
    pos_gmm->ComputeGconsts();
    neg_ngmm.CopyToDiagGmm(neg_gmm);
    neg_gmm->ComputeGconsts();
  }
}

void PosNegBL::ConvertPosNegGaussianToNNLayer(
    const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm,
    const Vector<double> &pos2neg_log_prior_ratio,
    const Vector<double> &var_scale,
    Matrix<BaseFloat> &linearity,
    Vector<BaseFloat> &bias) {
  if (linearity.NumRows() != pos_am_gmm.NumPdfs()
      || linearity.NumCols() != pos_am_gmm.Dim()) {
    linearity.Resize(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
  }
  if (bias.Dim() != pos_am_gmm.NumPdfs()) {
    bias.Resize(pos_am_gmm.NumPdfs(), kSetZero);
  }

  Vector<double> mu(pos_am_gmm.Dim(), kSetZero);

  int32 num_pdf = pos_am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {

    // iterate all the Gaussians
    const DiagGmm *gmm_pos = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm_pos(*gmm_pos);

    const DiagGmm *gmm_neg = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm_neg(*gmm_neg);

    KALDI_ASSERT(gmm_pos->NumGauss()==1 && gmm_neg->NumGauss()==1);

    mu.CopyRowFromMat(ngmm_pos.means_, 0);  // pos_mu
    mu.AddVec(-1.0, ngmm_neg.means_.Row(0));  // pos_mu - neg_mu
    mu.DivElements(ngmm_pos.vars_.Row(0));  // (pos_mu - neg_mu)_T / var
    mu.Scale(var_scale(pdf));

    linearity.CopyRowFromVec(Vector<BaseFloat>(mu), pdf);

    mu.SetZero();
    mu.AddVec2(1.0, ngmm_pos.means_.Row(0));  // pos_mu * pos_mu
    mu.AddVec2(-1.0, ngmm_neg.means_.Row(0));  // pos_mu * pos_mu - neg_mu * neg_mu
    mu.DivElements(ngmm_pos.vars_.Row(0));  // pos_mu * var_-1 * pos_mu_T - neg_mu * var_-1 * neg_mu_T

    bias(pdf) = static_cast<BaseFloat>(pos2neg_log_prior_ratio(pdf)
        - 0.5 * var_scale(pdf) * mu.Sum());
  }

}

}  // namespace kaldi
