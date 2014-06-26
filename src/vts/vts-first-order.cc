/*
 * vts-first-order.cc
 *
 *  Created on: Oct 20, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "decoder/decodable-am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "vts/vts-first-order.h"
#include "feat/feature-functions.h"

namespace kaldi {

// in-place change from original feature GMM to normalized feature GMM
void GmmToNormalizedGmm(const Vector<double> &mean, const Vector<double> &std,
                        AmDiagGmm &am_gmm) {

  Vector<double> inv_std(std);
  inv_std.InvertElements();
  Vector<double> inv_std2(inv_std);
  inv_std2.ApplyPow(2.0);

  // iterate all the GMMs
  int32 num_pdf = am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *gmm = &(am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm(*gmm);

    ngmm.means_.AddVecToRows(-1.0, mean);  // m_x - mu_x
    ngmm.means_.MulColsVec(inv_std);  // (m_x - mu_x) / std_x

    ngmm.vars_.MulColsVec(inv_std2);  // v_y = v_x / (std_x * std_x)

    ngmm.CopyToDiagGmm(gmm);
    gmm->ComputeGconsts();
  }
}

// in-place change from normalized feature GMM to original feature GMM
void NormalizedGmmToGmm(const Vector<double> &mean, const Vector<double> &std,
                        AmDiagGmm &am_gmm) {

  Vector<double> std2(std);
  std2.ApplyPow(2.0);

  // iterate all the GMMs
  int32 num_pdf = am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *gmm = &(am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm(*gmm);

    ngmm.means_.MulColsVec(std);  // m_y * std_x
    ngmm.means_.AddVecToRows(1.0, mean);  // m_y * std_x + mu_x

    ngmm.vars_.MulColsVec(std2);  // v_y * (std_x * std_x)

    ngmm.CopyToDiagGmm(gmm);
    gmm->ComputeGconsts();
  }
}

/*
 * Compute the KL-divergence of two diagonal Gaussians.
 * Refer to: http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 * P represents the true distribution, i.e. the true model
 * Q represents the estimated distribution, i.e. the estimation from samples
 *
 * KL(P||Q) = sum( ln(p(x)/q(x)) * p(x) dx )
 *
 * For multivariate Gaussian
 * KL(P||Q) = 0.5 * ( tr(Sigma_Q^{-1} * Sigma_P)
 *    + (mu_Q - mu_P)^T * Sigma_Q^{-1} * (mu_Q - mu_P)
 *    - ln( det(Sigma_P) / det(Sigma_Q) ) - D )
 *
 *    where D is the dimension of the feature vector.
 *
 */
double KLDivergenceDiagGaussian(const Vector<double> &p_mean,
                                const Vector<double> &p_var,
                                const Vector<double> &q_mean,
                                const Vector<double> &q_var) {
  double kl = -p_mean.Dim();
  for (int32 i = 0; i < p_mean.Dim(); ++i) {
    kl += (p_var(i) / q_var(i)
        + (q_mean(i) - p_mean(i)) * (q_mean(i) - p_mean(i)) / q_var(i)
        - log(p_var(i)) + log(q_var(i)));
  }
  kl *= 0.5;

  return kl;
}

/*
 * Generate the DCT and inverse DCT transforms with/without the Cepstral liftering.
 */
void GenerateDCTmatrix(int32 num_cepstral,
                       int32 num_fbank,
                       BaseFloat ceplifter,
                       Matrix<double> *dct_mat,
                       Matrix<double> *inv_dct_mat) {
  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG<< "Generate DCT matrics ...";
  }

  // compute the DCT matrix, in HTK C0 is the last component, i.e. C1, C2, ..., C12, C0
  // while in Kaldi, the order is C0, C1, C2, ..., C12, we will comply to the Kaldi style

  if (dct_mat->NumRows() != num_cepstral || dct_mat->NumCols() != num_fbank) {
    dct_mat->Resize(num_cepstral, num_fbank, kSetZero);
  } else {
    dct_mat->SetZero();
  }
  ComputeDctMatrix(dct_mat);

  Vector<double> ceplifter_vec(num_cepstral, kSetZero);
  if(ceplifter > 0) {
    ComputeLifterCoeffs(ceplifter, &ceplifter_vec);
  }

  if (g_kaldi_verbose_level >= 2) {
    KALDI_LOG << "DCT Transform: " << *dct_mat;
    KALDI_LOG << "Cepstal Lifter Vector: " << ceplifter_vec;
  }

  // generate the inverse transforms, pinv(C)=C_T * inv(C * C_T)
  if (inv_dct_mat->NumRows() != num_fbank
      || inv_dct_mat->NumCols() != num_cepstral) {
    inv_dct_mat->Resize(num_fbank, num_cepstral, kSetZero);
  } else {
    inv_dct_mat->SetZero();
  }

  Matrix<double> c_ct(num_cepstral, num_cepstral);
  c_ct.AddMatMat(1.0, *dct_mat, kNoTrans, *dct_mat, kTrans, 0.0);
  c_ct.Invert();
  inv_dct_mat->AddMatMat(1.0, *dct_mat, kTrans, c_ct, kNoTrans, 0.0);

  if (g_kaldi_verbose_level >= 2) {
    KALDI_LOG << "Inverse DCT: " << *inv_dct_mat;
  }

  // multiply the ceplifter coefficients to the DCT and inverse DCT matrix for simplicity.
  // i.e. set C=L*C, inv_C = inv_C * inv_L
  // as L is diagonal matrix, the multiplication is simply the scaling of each row
  if (ceplifter > 0) {
    dct_mat->MulRowsVec(ceplifter_vec);
    ceplifter_vec.InvertElements();
    inv_dct_mat->MulColsVec(ceplifter_vec);

    if (g_kaldi_verbose_level >= 2) {
      KALDI_LOG << "DCT Transform with CepLifter: " << *dct_mat;
      KALDI_LOG << "Inverse DCT Transform with CepLifter: " << *inv_dct_mat;
    }
  }

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG << "DCT matrices generation done!";
  }
}

/*
 * Get the global index of the first Gaussian of a given pdf.
 */
int32 GetGaussianOffset(const AmDiagGmm &am_gmm, int32 pdf_id) {
  int32 idx = 0;
  for (int32 i = 0; i < pdf_id && i < am_gmm.NumPdfs(); ++i) {
    idx += am_gmm.NumGaussInPdf(i);
  }
  return idx;
}

void EstimateInitialNoiseModel(const Matrix<BaseFloat> &features,
                               int32 feat_dim,
                               int32 num_static,
                               int32 noise_frames,
                               bool zero_mu_z_deltas,
                               Vector<double> *mu_h,
                               Vector<double> *mu_z,
                               Vector<double> *var_z) {
  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG<< "Estimating initial noise model...";
  }

  KALDI_ASSERT((mu_h->Dim() == feat_dim) && (mu_z->Dim() == feat_dim)
      && (var_z->Dim() == feat_dim));
  mu_h->SetZero();  // the convolutional noise, initialized to be 0
  mu_z->SetZero();// the additive noise mean, only static part to be estimated
  var_z->SetZero();// the additive noise covariance, diagonal, to be estimated

  Vector<double> mean_obs(feat_dim);// mean of
  Vector<double> x2(feat_dim);// x^2 stats
  int32 i, tot_frames;
  // accumulate the starting stats
  for (i = 0; i < noise_frames && i < features.NumRows(); ++i) {
    mu_z->AddVec(1.0, features.Row(i));
    var_z->AddVec2(1.0, features.Row(i));
  }
  tot_frames = i;
  // accumulate the ending stats
  i = features.NumRows() - noise_frames;
  if (i < 0) i = 0;
  for (; i < features.NumRows(); ++i, ++tot_frames) {
    mu_z->AddVec(1.0, features.Row(i));
    var_z->AddVec2(1.0, features.Row(i));
  }
  mu_z->Scale(1.0 / tot_frames);
  var_z->Scale(1.0 / tot_frames);
  var_z->AddVec2(-1.0, *mu_z);

  ///////////////////////////////////////////////////
  // As we assume the additive noise has 0 delta and delta-delta mean
  if(zero_mu_z_deltas) {
    for (i = num_static; i < feat_dim; ++i) {
      (*mu_z)(i) = 0.0;
    }
  }

  ///////////////////////////////////////////////////
  // As we assume the additive noise only, the convoluational noise, mean_h, is set to 0
  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG << "Initial model estimation done!";
  }
}

/*
 * Compensate a  Diagonal Gaussian using estimated noise parameters.
 *
 * mean is the clean mean to be compensated;
 * cov is the diagonal elements for the Diagonal Gaussian covariance with clean
 * values to be compensated.
 *
 * Matrix Jx and Jz are used to keep gradients.
 *
 */
void CompensateDiagGaussian(const Vector<double> &mu_h,
                            const Vector<double> &mu_z,
                            const Vector<double> &var_z,
                            int32 num_cepstral,
                            int32 num_fbank,
                            const Matrix<double> &dct_mat,
                            const Matrix<double> &inv_dct_mat,
                            Vector<double> &mean,
                            Vector<double> &cov,
                            Matrix<double> &Jx,
                            Matrix<double> &Jz) {
  // compute the necessary transforms
  Vector<double> mu_y_s(num_cepstral);
  Vector<double> tmp_fbank(num_fbank);

  Jx.Resize(num_cepstral, num_cepstral, kSetZero);
  Jz.Resize(num_cepstral, num_cepstral, kSetZero);

  for (int32 ii = 0; ii < num_cepstral; ++ii) {
    mu_y_s(ii) = mu_z(ii) - mean(ii) - mu_h(ii);
  }  // mu_n - mu_x - mu_h
  tmp_fbank.AddMatVec(1.0, inv_dct_mat, kNoTrans, mu_y_s, 0.0);  // C_inv * (mu_n - mu_x - mu_h)
  tmp_fbank.ApplyExp();  // exp( C_inv * (mu_n - mu_x - mu_h) )
  tmp_fbank.Add(1.0);  // 1 + exp( C_inv * (mu_n - mu_x - mu_h) )
  Vector<double> tmp_inv(tmp_fbank);  // keep a version
  tmp_fbank.ApplyLog();  // log ( 1 + exp( C_inv * (mu_n - mu_x - mu_h) ) )
  tmp_inv.InvertElements();  // 1.0 / ( 1 + exp( C_inv * (mu_n - mu_x - mu_h) ) )

  // new static mean
  for (int32 ii = 0; ii < num_cepstral; ++ii) {
    mu_y_s(ii) = mean(ii) + mu_h(ii);
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
    KALDI_LOG<< "Mean Before: " << mean;
  }
  Vector<double> tmp_mu(num_cepstral);
  SubVector<double> mu_s(mean, 0, num_cepstral);
  mu_s.CopyFromVec(mu_y_s);
  SubVector<double> mu_dt(mean, num_cepstral, num_cepstral);
  tmp_mu.CopyFromVec(mu_dt);
  mu_dt.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
  SubVector<double> mu_acc(mean, 2 * num_cepstral, num_cepstral);
  tmp_mu.CopyFromVec(mu_acc);
  mu_acc.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<< "Mean After: " << mean;
  }

  // compute and update covariance
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<< "Covarianc Before: " << cov;
  }
  for (int32 ii = 0; ii < 3; ++ii) {
    Matrix<double> tmp_var1(Jx), tmp_var2(Jz), new_var(num_cepstral,
                                                       num_cepstral);
    SubVector<double> x_var(cov, ii * num_cepstral, num_cepstral);
    SubVector<double> n_var(var_z, ii * num_cepstral, num_cepstral);

    tmp_var1.MulColsVec(x_var);
    new_var.AddMatMat(1.0, tmp_var1, kNoTrans, Jx, kTrans, 0.0);

    tmp_var2.MulColsVec(n_var);
    new_var.AddMatMat(1.0, tmp_var2, kNoTrans, Jz, kTrans, 1.0);

    x_var.CopyDiagFromMat(new_var);
  }
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<< "Covariance After: " << cov;
  }
}

/*
 * Do the compensation using the current noise model parameters for a diagonal GMM.
 * Also keep the statistics of the Jx, and Jz for next iteration of noise estimation.
 *
 * noise_gmm is initialized as the clean model.
 *
 */
void CompensateDiagGmm(const Vector<double> &mu_h,
                       const Vector<double> &mu_z,
                       const Vector<double> &var_z,
                       int32 num_cepstral,
                       int32 num_fbank,
                       const Matrix<double> &dct_mat,
                       const Matrix<double> &inv_dct_mat,
                       DiagGmm &noise_gmm,
                       std::vector<Matrix<double> > &Jx,
                       std::vector<Matrix<double> > &Jz) {

//KALDI_LOG << "Beginning compensate model ...";

// iterate all the Gaussians
  DiagGmmNormal ngmm(noise_gmm);

  int32 num_gauss = noise_gmm.NumGauss();
  for (int32 g = 0; g < num_gauss; ++g) {

    //KALDI_LOG << "pdf_id: " << pdf << ", gauss_id: " << g;

    Vector<double> cur_mean(ngmm.means_.Row(g));
    Vector<double> cur_var(ngmm.vars_.Row(g));
    CompensateDiagGaussian(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat,
                           inv_dct_mat,
                           cur_mean, cur_var, Jx[g], Jz[g]);
    ngmm.means_.CopyRowFromVec(cur_mean, g);
    ngmm.vars_.CopyRowFromVec(cur_var, g);
  }

  ngmm.CopyToDiagGmm(&noise_gmm);
  noise_gmm.ComputeGconsts();

//KALDI_LOG << "Model compensation done!";
}

/*
 * Do the compensation using the current noise model parameters.
 * Also keep the statistics of the Jx, and Jz for next iteration of noise estimation.
 *
 * noise_am_gmm is initialized as the clean model.
 *
 */
void CompensateModel(const Vector<double> &mu_h,
                     const Vector<double> &mu_z,
                     const Vector<double> &var_z,
                     int32 num_cepstral,
                     int32 num_fbank,
                     const Matrix<double> &dct_mat,
                     const Matrix<double> &inv_dct_mat,
                     AmDiagGmm &noise_am_gmm,
                     std::vector<Matrix<double> > &Jx,
                     std::vector<Matrix<double> > &Jz) {

  //KALDI_LOG << "Beginning compensate model ...";

  // iterate all the GMMs
  int32 num_pdf = noise_am_gmm.NumPdfs();
  for (int32 pdf = 0, tot_gauss_id = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *gmm = &(noise_am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm(*gmm);

    int32 num_gauss = gmm->NumGauss();
    for (int32 g = 0; g < num_gauss; ++g, ++tot_gauss_id) {

      //KALDI_LOG << "pdf_id: " << pdf << ", gauss_id: " << g;

      Vector<double> cur_mean(ngmm.means_.Row(g));
      Vector<double> cur_var(ngmm.vars_.Row(g));
      CompensateDiagGaussian(mu_h, mu_z, var_z, num_cepstral, num_fbank,
                             dct_mat, inv_dct_mat, cur_mean, cur_var,
                             Jx[tot_gauss_id],
                             Jz[tot_gauss_id]);
      ngmm.means_.CopyRowFromVec(cur_mean, g);
      ngmm.vars_.CopyRowFromVec(cur_var, g);
    }

    ngmm.CopyToDiagGmm(gmm);
    gmm->ComputeGconsts();
  }

  //KALDI_LOG << "Model compensation done!";
}

/*
 * gamma, gamma_p, gamma_q must be initialized to be zeros vectors/matrices.
 *
 */
BaseFloat AccumulatePosteriorStatistics(const AmDiagGmm &am_gmm,
                                        const TransitionModel &trans_model,
                                        const std::vector<int32> &alignment,
                                        const Matrix<BaseFloat> &features,
                                        Vector<double> &gamma,
                                        Matrix<double> &gamma_p,
                                        Matrix<double> &gamma_q) {
  KALDI_ASSERT(alignment.size() == features.NumRows());
  gamma.SetZero();
  gamma_p.SetZero();
  gamma_q.SetZero();

  BaseFloat like = 0.0;
  for (size_t t = 0; t < alignment.size(); t++) {
    int32 tid = alignment[t];
    int32 pdf_id = trans_model.TransitionIdToPdf(tid);

    // given the alignment, at each time t, the state posterior is one for the
    // aligned state, 0 for all the others
    const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
    Vector<BaseFloat> this_post_vec;
    like += gmm.ComponentPosteriors(features.Row(t), &this_post_vec);

    if (!this_post_vec.IsZero()) {
      int32 gauss_offset = GetGaussianOffset(am_gmm, pdf_id);
      for (int32 m = 0; m < gmm.NumGauss(); ++m) {
        gamma(gauss_offset + m) += this_post_vec(m);
        (gamma_p.Row(gauss_offset + m)).AddVec(this_post_vec(m),
                                               features.Row(t));
        (gamma_q.Row(gauss_offset + m)).AddVec2(this_post_vec(m),
                                                features.Row(t));
      }
    }
  }

  return like;
}

/*
 * Compute the model likelihood of given feature and alignment.
 *
 */
BaseFloat ComputeLogLikelihood(const AmDiagGmm &am_gmm,
                               const TransitionModel &trans_model,
                               const std::vector<int32> &alignment,
                               const Matrix<BaseFloat> &features) {
  KALDI_ASSERT(alignment.size() == features.NumRows());
  BaseFloat like = 0.0;
  for (size_t t = 0; t < alignment.size(); t++) {
    int32 tid = alignment[t];
    int32 pdf_id = trans_model.TransitionIdToPdf(tid);

    // given the alignment, at each time t, the state posterior is one for the
    // aligned state, 0 for all the others
    const DiagGmm &gmm = am_gmm.GetPdf(pdf_id);
    like += gmm.LogLikelihood(features.Row(t));
  }

  return like;
}

/*
 * Back off if the new estimation doesn't increase the likelihood.
 *
 * The input noisy_am_gmm is the clean model compensated by mu_h0, mu_z0, var_z0.
 * After this function, it will be updated with the new noise estimation.
 *
 * If the estimation completely revert back to the original estimation, then return false;
 *
 */
bool BackOff(const AmDiagGmm &clean_am_gmm, const TransitionModel &trans_model,
             const std::vector<int32> &alignment,
             const Matrix<BaseFloat> &features,
             int32 num_cepstral, int32 num_fbank,
             const Matrix<double> &dct_mat,
             const Matrix<double> &inv_dct_mat,
             const Vector<double> &mu_h0,
             const Vector<double> &mu_z0,
             const Vector<double> &var_z0,
             Vector<double> &mu_h,
             bool update_mu_h, Vector<double> &mu_z,
             bool update_mu_z,
             Vector<double> &var_z, bool update_var_z,
             AmDiagGmm &noise_am_gmm,
             std::vector<Matrix<double> > &Jx,
             std::vector<Matrix<double> > &Jz) {

  bool new_estimate = true;

  BaseFloat pre_log_likes = ComputeLogLikelihood(noise_am_gmm, trans_model,
                                                 alignment,
                                                 features);
  BaseFloat cur_log_likes, ratio = 1.0, delta = 0.5;
  cur_log_likes = pre_log_likes;
  Vector<double> cur_mu_h(mu_h), cur_mu_z(mu_z), cur_var_z(var_z);

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG<< "Begining back-off: [" << (update_mu_h ? "mu_h, " : "")
    << (update_mu_z ? "mu_z, " : "") << (update_var_z ? "var_z, " : "")
    << "]";
    KALDI_LOG << "Initial Log Likelihood: " << pre_log_likes;
    /*KALDI_LOG << "Initial mu_h: " << mu_h0;
     KALDI_LOG << "Initial mu_z: " << mu_z0;
     KALDI_LOG << "Initial var_z: " << var_z0;*/
  }

  do {

    // interpolate
    if (update_mu_h) {
      cur_mu_h.CopyFromVec(mu_h);
      cur_mu_h.Scale(ratio);
      cur_mu_h.AddVec(1 - ratio, mu_h0);  // r * mu_h + (1-r) * mu_h0
    }

    if (update_mu_z) {
      cur_mu_z.CopyFromVec(mu_z);
      cur_mu_z.Scale(ratio);
      cur_mu_z.AddVec(1 - ratio, mu_z0);
    }

    if (update_var_z) {
      cur_var_z.CopyFromVec(var_z);
      cur_var_z.Scale(ratio);
      cur_var_z.AddVec(1 - ratio, var_z0);
    }

    if (g_kaldi_verbose_level >= 2) {
      KALDI_LOG<< "Current mu_h: " << cur_mu_h;
      KALDI_LOG << "Current mu_z: " << cur_mu_z;
      KALDI_LOG << "Current var_z: " << cur_var_z;
    }

    // re-initialize it with clean model
    noise_am_gmm.CopyFromAmDiagGmm(clean_am_gmm);

    CompensateModel(cur_mu_h, cur_mu_z, cur_var_z, num_cepstral, num_fbank,
                    dct_mat,
                    inv_dct_mat, noise_am_gmm, Jx, Jz);

    cur_log_likes = ComputeLogLikelihood(noise_am_gmm, trans_model, alignment,
                                         features);

    if (g_kaldi_verbose_level >= 1) {
      KALDI_LOG<< "New estimation ratio: " << ratio << ", Log Likelihood: "
          << cur_log_likes;
    }

    ratio *= delta;
    if (ratio < 1e-3) {
      ratio = 0.0;  // if the ratio is too small, we simply revert back to the original estimation
      new_estimate = false;
    }

  } while (cur_log_likes < pre_log_likes);

  mu_h.CopyFromVec(cur_mu_h);
  mu_z.CopyFromVec(cur_mu_z);
  var_z.CopyFromVec(cur_var_z);

  return new_estimate;

}

/*
 * The AM_GMM is the noise compensated model using the existing noise estimation.
 *
 * The mu_h, mu_z and var_z contains the current estimation and will be updated in this function.
 *
 * Cambridge implementation.
 */

void EstimateStaticNoiseMean(const AmDiagGmm &noise_am_gmm,
                             const Vector<double> &gamma,
                             const Matrix<double> &gamma_p,
                             const Matrix<double> &gamma_q,
                             const std::vector<Matrix<double> > &Jx,
                             const std::vector<Matrix<double> > &Jz,
                             int32 num_cepstral, BaseFloat max_magnitude,
                             SubVector<double> &mu_h_s,
                             SubVector<double> &mu_z_s) {

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG<< "Estimate static noise mean staring ...";
  }

  // Limit the change of the mean
  double stigma = static_cast<double>(max_magnitude);
  bool update = true;

  Vector<double> d(num_cepstral), u(num_cepstral);
  Matrix<double> E(num_cepstral, num_cepstral), F(num_cepstral, num_cepstral);
  Matrix<double> V(num_cepstral, num_cepstral), W(num_cepstral, num_cepstral);

  // iterate all the Gaussians in the HMM
  for (int32 pdf_id = 0, tot_gauss_id = 0; pdf_id < noise_am_gmm.NumPdfs();
      ++pdf_id) {
    const DiagGmm *gmm = &(noise_am_gmm.GetPdf(pdf_id));
    DiagGmmNormal ngmm(*gmm);

    int32 num_gauss = gmm->NumGauss();
    for (int32 g = 0; g < num_gauss; ++g, ++tot_gauss_id) {

      // only when the posterior is above 0, will it affect the estimation
      if (gamma(tot_gauss_id) > 0.0) {
        Vector<double> g_mu(
            SubVector<double>(ngmm.means_.Row(g), 0, num_cepstral));  // f(mu_x, mu_h, mu_z)
        Vector<double> g_var(
            SubVector<double>(ngmm.vars_.Row(g), 0, num_cepstral));
        g_var.InvertElements();// inv_var_y

        g_mu.AddMatVec(-1.0, Jx[tot_gauss_id], kNoTrans, mu_h_s, 1.0);// f(mu_x, mu_h, mu_z) - Jx * mu_h
        g_mu.AddMatVec(-1.0, Jz[tot_gauss_id], kNoTrans, mu_z_s, 1.0);// f(mu_x, mu_h, mu_z) - Jx * mu_h - Jz * mu_z
        g_mu.Scale(-1.0 * gamma(tot_gauss_id));// - [ f(mu_x, mu_h, mu_z) - Jx * mu_h - Jz * mu_z ] * gamma
        g_mu.AddVec(
            1.0, SubVector<double>(gamma_p.Row(tot_gauss_id), 0, num_cepstral));// gamma_p - [ f(mu_x, mu_h, mu_z) - Jx * mu_h - Jz * mu_z ] * gamma

        Matrix<double> tmp_x(Jx[tot_gauss_id], kTrans),// Jx_T
        tmp_z(Jz[tot_gauss_id], kTrans);// Jz_T
        tmp_x.MulColsVec(g_var);// Jx_T * inv_var_y
        tmp_z.MulColsVec(g_var);// Jz_T * inv_var_y

        d.AddMatVec(1.0, tmp_z, kNoTrans, g_mu, 1.0);// Jz_T * inv_var_y * { gamma_p - [ f(mu_x, mu_h, mu_z) - Jx * mu_h - Jz * mu_z ] * gamma }
        E.AddMatMat(gamma(tot_gauss_id), tmp_z, kNoTrans, Jz[tot_gauss_id],
            kNoTrans, 1.0);// Jz_T * inv_var_y * Jz * gamma
        F.AddMatMat(gamma(tot_gauss_id), tmp_z, kNoTrans, Jx[tot_gauss_id],
            kNoTrans, 1.0);// Jz_T * inv_var_y * Jx * gamma

        u.AddMatVec(1.0, tmp_x, kNoTrans, g_mu, 1.0);// Jx_T * inv_var_y * { gamma_p - [ f(mu_x, mu_h, mu_z) - Jx * mu_h - Jz * mu_z ] * gamma }
        V.AddMatMat(gamma(tot_gauss_id), tmp_x, kNoTrans, Jz[tot_gauss_id],
            kNoTrans, 1.0);// Jx_T * inv_var_y * Jz * gamma
        W.AddMatMat(gamma(tot_gauss_id), tmp_x, kNoTrans, Jx[tot_gauss_id],
            kNoTrans, 1.0);// Jx_T * inv_var_y * Jx * gamma
      }
    }
  }

  Vector<double> vec1(num_cepstral), vec2(num_cepstral);
  Matrix<double> inv_mat(num_cepstral, num_cepstral), mat1(num_cepstral,
      num_cepstral), mat2(
      num_cepstral, num_cepstral);

  // compute new estimate of mu_z_s
  inv_mat.CopyFromMat(F);
  inv_mat.Invert();// inv_F
  mat1.AddMatMat(1.0, W, kNoTrans, inv_mat, kNoTrans, 0.0);// W * inv_F
  mat2.CopyFromMat(V);
  mat2.AddMatMat(-1.0, mat1, kNoTrans, E, kNoTrans, 1.0);// V - W * inv_F * E
  mat2.Invert();// inv(V - W * inv_F * E)
  vec1.CopyFromVec(u);
  vec1.AddMatVec(-1.0, mat1, kNoTrans, d, 1.0);// u - W * inv_F * d
  vec2.AddMatVec(1.0, mat2, kNoTrans, vec1, 0.0);// inv(V - W * inv_F * E) * ( u - W * inv_F * d )
  // limit the update amount
  update = true;
  for (int32 i = 0; update && i < num_cepstral; ++i) {
    if (vec2(i) > stigma || vec2(i) < -stigma) {
      update = false;
    }
  }
  if (update) mu_z_s.CopyFromVec(vec2);

  // compute new estimate of mu_h_s
  /*
   * As inv(F - E * inv_V * W) = inv(V - W * inv_F * E)_T, no need to recompute them.
   *
   inv_mat.CopyFromMat(V);
   inv_mat.Invert();// inv_V
   mat1.AddMatMat(1.0, E, kNoTrans, inv_mat, kNoTrans, 0.0);// E * inv_V
   mat2.CopyFromMat(F);
   mat2.AddMatMat(-1.0, mat1, kNoTrans, W, kNoTrans, 1.0);// F - E * inv_V * W
   mat2.Invert();// inv(F - E * inv_V * W)*/
  mat1.AddMatMat(1.0, E, kNoTrans, inv_mat, kTrans, 0.0);  // E * inv_V = E * inv_F_T
  vec1.CopyFromVec(d);
  vec1.AddMatVec(-1.0, mat1, kNoTrans, u, 1.0);// d - E * inv_V * u
  vec2.AddMatVec(1.0, mat2, kTrans, vec1, 0.0);// inv(F - E * inv_V * W) * ( d - E * inv_V * u ) = inv(V - W * inv_F * E)_T * ( d - E * inv_V * u )
  // limit the update amount
  update = true;
  for (int32 i = 0; update && i < num_cepstral; ++i) {
    if (vec2(i) > stigma || vec2(i) < -stigma) {
      update = false;
    }
  }
  if (update) mu_h_s.CopyFromVec(vec2);

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG << "Estimate static noise mean done!";
  }

}

/*
 * Estimate the Noise Variance
 *
 * Refer to the paper "Noise Adaptive Training for Robust
 * Automatic Speech Recognition", Alex Acero, Microsoft
 *
 */
void EstimateAdditiveNoiseVariance(const AmDiagGmm &noise_am_gmm,
                                   const Vector<double> &gamma,
                                   const Matrix<double> &gamma_p,
                                   const Matrix<double> &gamma_q,
                                   const std::vector<Matrix<double> > &Jz,
                                   int32 num_cepstral,
                                   int32 feat_dim,
                                   BaseFloat lrate,
                                   Vector<double> &var_z) {

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG<< "Estimate additive noise covariance staring ...";
  }

  // Diagonal loading to stabilize the Hessian
  // Refer to "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"
  double epsilon = 1.0;
  // Limit the change of the variance
  double stigma = 1.0;

  Vector<double> dt1(feat_dim, kSetZero);// Jacobian is still a vector
  Matrix<double> dt2(feat_dim, feat_dim, kSetZero);// Hessian is a matrix

  Vector<double> sum_sm1(feat_dim, kSetZero);// the accumulated statistics for Jacobian of all the Gaussians
  Vector<double> sum_sm2(feat_dim, kSetZero);// the accumulated statistics for Hessian of all the Gaussians

  // iterate all the Gaussians in the HMM
  for (int32 pdf_id = 0, tot_gauss_id = 0; pdf_id < noise_am_gmm.NumPdfs();
      ++pdf_id) {
    const DiagGmm *gmm = &(noise_am_gmm.GetPdf(pdf_id));
    DiagGmmNormal ngmm(*gmm);

    int32 num_gauss = gmm->NumGauss();
    for (int32 g = 0; g < num_gauss; ++g, ++tot_gauss_id) {

      // only when the posterior is above 0, will it affect the estimation
      if (gamma(tot_gauss_id) > 0.0) {
        Matrix<double> Jz2(Jz[tot_gauss_id]);
        Jz2.ApplyPow(2.0);  // Elementwise square of Jz
        Jz2.Transpose();

        // first compute the shared term
        Vector<double> share_sm(feat_dim, kSetZero);
        share_sm.AddVec2(-1 * gamma(tot_gauss_id), ngmm.means_.Row(g));// - mu_y .* mu_y * gamma
        share_sm.AddVec(-1.0, gamma_q.Row(tot_gauss_id));// - mu_y .* mu_y * gamma - gamma_q
        share_sm.AddVecVec(2.0, ngmm.means_.Row(g), gamma_p.Row(tot_gauss_id),
            1.0);// - mu_y .* mu_y * gamma - gamma_q + 2 * mu_y .* gamma_p

        Vector<double> cur_sm(feat_dim, kSetZero);
        Vector<double> cur_var(ngmm.vars_.Row(g));
        // Jacobian
        cur_sm.AddVec(gamma(tot_gauss_id), ngmm.vars_.Row(g));// var_y * gamma
        cur_sm.AddVec(1.0, share_sm);// var_y * gamma - mu_y .* mu_y * gamma - gamma_q + 2 * mu_y .* gamma_p
        cur_var.ApplyPow(2.0);
        Vector<double> sm1(feat_dim, kSetZero);
        sm1.AddVecDivVec(1.0, cur_sm, cur_var, 1.0);// cur_sm / ( var_y * var_y )

        for (int32 p = 0; p < feat_dim; ++p) {
          int32 c = p / num_cepstral;
          int32 r = p % num_cepstral;
          SubVector<double> jzp(Jz2, r);  // Jz2 are transposed
          SubVector<double> vsm(sm1, c * num_cepstral, num_cepstral);
          Vector<double> product(num_cepstral, kSetZero);
          product.AddVecVec(1.0, jzp, vsm, 0.0);
          dt1(p) += product.Sum();
        }

        // Hessian
        cur_sm.SetZero();
        cur_sm.AddVec(0.5 * gamma(tot_gauss_id), ngmm.vars_.Row(g));// 0.5 * var_y * gamma
        cur_sm.AddVec(1.0, share_sm);// 0.5 * var_y * gamma - mu_y .* mu_y * gamma - gamma_q + 2 * mu_y .* gamma_p
        cur_var.CopyFromVec(ngmm.vars_.Row(g));
        cur_var.ApplyPow(3.0);
        Vector<double> sm2(feat_dim, kSetZero);
        sm2.AddVecDivVec(1.0, cur_sm, cur_var, 1.0);// cur_sm / ( var_y * var_y * var_y )

        for (int32 c = 0; c < feat_dim / num_cepstral; ++c) {
          SubVector<double> vsm(sm2, c * num_cepstral, num_cepstral);
          for (int32 pr = 0; pr < num_cepstral; ++pr) {
            SubVector<double> jzp(Jz2, pr);
            Vector<double> prod_p(num_cepstral, kSetZero);
            prod_p.AddVecVec(1.0, jzp, vsm, 0.0);
            for (int32 lr = 0; lr < num_cepstral; ++lr) {
              SubVector<double> jzl(Jz2, lr);
              Vector<double> product(num_cepstral, kSetZero);
              product.AddVecVec(1.0, jzl, prod_p, 0.0);
              dt2(c * num_cepstral + pr, c * num_cepstral + lr) +=
              product.Sum();
            }
          }
        }
      }
    }
  }

  dt1.MulElements(var_z);
  dt1.Scale(-0.5);

  dt2.MulColsVec(var_z);
  dt2.MulRowsVec(var_z);
  for (int32 i = 0; i < feat_dim; ++i) {
    dt2(i, i) += (dt1(i) - epsilon);
  }
  dt2.Invert();
  Vector<double> grad(feat_dim, kSetZero);
  grad.AddMatVec(1.0, dt2, kNoTrans, dt1, 0.0);
  grad.Scale(lrate);
  for (int32 i = 0; i < feat_dim; ++i) {
    if (grad(i) > stigma) grad(i) = stigma;
    if (grad(i) < -stigma) grad(i) = -stigma;
  }
  //KALDI_LOG << "gradient: " << grad;
  grad.ApplyExp();
  grad.InvertElements();
  var_z.MulElements(grad);

  if (g_kaldi_verbose_level >= 1) {
    KALDI_LOG << "Estimate additive noise covariance done!";
  }
}

void CompensateMultiFrameGmm(const Vector<double> &mu_h,
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
        for (int32 ii = 0; ii < num_cepstral; ++ii) {
          Jz(ii, ii) = 1.0 - Jz(ii, ii);
        }

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

void CompensateDiagGaussian_FBank(const Vector<double> &mu_h,
                                  const Vector<double> &mu_z,
                                  const Vector<double> &var_z,
                                  bool have_energy,
                                  int32 num_fbank,
                                  Vector<double> &mean,
                                  Vector<double> &cov,
                                  Matrix<double> &Jx,
                                  Matrix<double> &Jz) {
  // compute the necessary transforms
  Vector<double> mu_y_s(num_fbank);
  Vector<double> tmp_fbank(num_fbank);

  Jx.Resize(num_fbank, num_fbank, kSetZero);
  Jz.Resize(num_fbank, num_fbank, kSetZero);

  for (int32 ii = 0; ii < num_fbank; ++ii) {
    tmp_fbank(ii) = mu_z(ii) - mean(ii) - mu_h(ii);
  }  // mu_n - mu_x - mu_h
  tmp_fbank.ApplyExp();  // exp( mu_n - mu_x - mu_h )
  tmp_fbank.Add(1.0);  // 1 + exp( (mu_n - mu_x - mu_h) )
  Vector<double> tmp_inv(tmp_fbank);  // keep a version
  tmp_fbank.ApplyLog();  // log ( 1 + exp( (mu_n - mu_x - mu_h) ) )
  tmp_inv.InvertElements();  // 1.0 / ( 1 + exp( (mu_n - mu_x - mu_h) ) )

  // new static mean
  for (int32 ii = 0; ii < num_fbank; ++ii) {
    mu_y_s(ii) = mean(ii) + mu_h(ii) + tmp_fbank(ii);
  }  // mu_x + mu_h + log ( 1 + exp( (mu_n - mu_x - mu_h) ) )

  // compute J
  Jx.CopyDiagFromVec(tmp_inv);

  // compute I_J
  Jz.CopyFromMat(Jx);
  for (int32 ii = 0; ii < num_fbank; ++ii) {
    Jz(ii, ii) = 1.0 - Jz(ii, ii);
  }

  // compute and update mean
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<< "Mean Before: " << mean;
  }
  Vector<double> tmp_mu(num_fbank);
  SubVector<double> mu_s(mean, 0, num_fbank);
  mu_s.CopyFromVec(mu_y_s);
  SubVector<double> mu_dt(mean, num_fbank + (have_energy ? 1 : 0), num_fbank);
  tmp_mu.CopyFromVec(mu_dt);
  mu_dt.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
  SubVector<double> mu_acc(mean, 2 * (num_fbank + (have_energy ? 1 : 0)),
                           num_fbank);
  tmp_mu.CopyFromVec(mu_acc);
  mu_acc.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<< "Mean After: " << mean;
  }

  // compute and update covariance
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<< "Covarianc Before: " << cov;
  }
  for (int32 ii = 0; ii < 3; ++ii) {
    Matrix<double> tmp_var1(Jx), tmp_var2(Jz), new_var(num_fbank, num_fbank);
    SubVector<double> x_var(cov, ii * (num_fbank + (have_energy ? 1 : 0)),
                            num_fbank);
    SubVector<double> n_var(var_z, ii * num_fbank, num_fbank);

    tmp_var1.MulColsVec(x_var);
    new_var.AddMatMat(1.0, tmp_var1, kNoTrans, Jx, kTrans, 0.0);

    tmp_var2.MulColsVec(n_var);
    new_var.AddMatMat(1.0, tmp_var2, kNoTrans, Jz, kTrans, 1.0);

    x_var.CopyDiagFromMat(new_var);
  }
  if (g_kaldi_verbose_level >= 9) {
    KALDI_LOG<<"Covariance After: " << cov;
  }
}

}

// End namespace
