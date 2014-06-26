/*
 * vts/vts-accum-diag-gmm.cc
 *
 *  Created on: Nov 23, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include <algorithm>  // for std::max
#include <string>
#include <vector>

#include "gmm/diag-gmm.h"
#include "vts/vts-accum-diag-gmm.h"

namespace kaldi {

void VtsAccumDiagGmm::Resize(int32 num_comp, int32 dim, int32 num_cepstral,
                             GmmFlagsType flags) {
  KALDI_ASSERT(
      num_comp > 0 && dim > 0 && dim % num_cepstral == 0 && dim / num_cepstral == 3);
  num_comp_ = num_comp;
  num_cepstral_ = num_cepstral;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
  occupancy_.Resize(num_comp);

  if (flags_ & kGmmMeans) {
    mu_vs_.Resize(num_comp, num_cepstral);
    mu_vd_.Resize(num_comp, num_cepstral);
    mu_va_.Resize(num_comp, num_cepstral);
    mu_ms_.Resize(num_comp * num_cepstral, num_cepstral);
    mu_md_.Resize(num_comp * num_cepstral, num_cepstral);
    mu_ma_.Resize(num_comp * num_cepstral, num_cepstral);
  } else {
    mu_vs_.Resize(0, 0);
    mu_vd_.Resize(0, 0);
    mu_va_.Resize(0, 0);
    mu_ms_.Resize(0, 0);
    mu_md_.Resize(0, 0);
    mu_ma_.Resize(0, 0);
  }

  if (flags_ & kGmmVariances) {
    var_js_.Resize(num_comp, num_cepstral);
    var_jd_.Resize(num_comp, num_cepstral);
    var_ja_.Resize(num_comp, num_cepstral);
    var_hs_.Resize(num_comp * num_cepstral, num_cepstral);
    var_hd_.Resize(num_comp * num_cepstral, num_cepstral);
    var_ha_.Resize(num_comp * num_cepstral, num_cepstral);
  } else {
    var_js_.Resize(0, 0);
    var_jd_.Resize(0, 0);
    var_ja_.Resize(0, 0);
    var_hs_.Resize(0, 0);
    var_hd_.Resize(0, 0);
    var_ha_.Resize(0, 0);
  }

}

void VtsAccumDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 dimension, num_cepstrals, num_components;
  GmmFlagsType flags;
  std::string token;

  ExpectToken(in_stream, binary, "<VTSGMMACCS>");
  ExpectToken(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dimension);
  ExpectToken(in_stream, binary, "<NUMCEPSTRALS>");
  ReadBasicType(in_stream, binary, &num_cepstrals);
  ExpectToken(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_components);
  ExpectToken(in_stream, binary, "<FLAGS>");
  ReadBasicType(in_stream, binary, &flags);

  if (add) {
    if ((NumGauss() != 0 || Dim() != 0 || NumCepstral() != 0 || Flags() != 0)) {
      if (num_components != NumGauss() || dimension != Dim()
          || num_cepstrals != NumCepstral() || flags != Flags()) {
        KALDI_ERR<< "MlEstimatediagGmm::Read, dimension or flags mismatch, "
        << NumGauss() << ", " << Dim() << ", " << NumCepstral() << ", "
        << GmmFlagsToString(Flags()) << " vs. " << num_components << ", "
        << dimension << ", " << num_cepstrals << ", "<< flags << " (mixing accs from different "
        << "models?";
      }
    } else {
      Resize(num_components, dimension, num_cepstrals, flags);
    }
  } else {
    Resize(num_components, dimension, num_cepstrals, flags);
  }

  ReadToken(in_stream, binary, &token);
  while (token != "</VTSGMMACCS>") {
    if (token == "<OCCUPANCY>") {
      occupancy_.Read(in_stream, binary, add);
    } else if (token == "<MUVS>") {
      mu_vs_.Read(in_stream, binary, add);
    } else if (token == "<MUVD>") {
      mu_vd_.Read(in_stream, binary, add);
    } else if (token == "<MUVA>") {
      mu_va_.Read(in_stream, binary, add);
    } else if (token == "<MUMS>") {
      mu_ms_.Read(in_stream, binary, add);
    } else if (token == "<MUMD>") {
      mu_md_.Read(in_stream, binary, add);
    } else if (token == "<MUMA>") {
      mu_ma_.Read(in_stream, binary, add);
    } else if (token == "<VARJS>") {
      var_js_.Read(in_stream, binary, add);
    } else if (token == "<VARJD>") {
      var_jd_.Read(in_stream, binary, add);
    } else if (token == "<VARJA>") {
      var_ja_.Read(in_stream, binary, add);
    } else if (token == "<VARHS>") {
      var_hs_.Read(in_stream, binary, add);
    } else if (token == "<VARHD>") {
      var_hd_.Read(in_stream, binary, add);
    } else if (token == "<VARHA>") {
      var_ha_.Read(in_stream, binary, add);
    } else {
      KALDI_ERR<< "Unexpected token '" << token << "' in model file ";
    }
    ReadToken(in_stream, binary, &token);
  }
}

void VtsAccumDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  WriteToken(out_stream, binary, "<VTSGMMACCS>");
  WriteToken(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteToken(out_stream, binary, "<NUMCEPSTRALS>");
  WriteBasicType(out_stream, binary, num_cepstral_);
  WriteToken(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteToken(out_stream, binary, "<FLAGS>");
  WriteBasicType(out_stream, binary, flags_);

  // convert into BaseFloat before writing things
  Vector<BaseFloat> occupancy_bf(occupancy_.Dim());
  Matrix<BaseFloat> vec_bf(num_comp_, num_cepstral_);
  Matrix<BaseFloat> mat_bf(num_comp_ * num_cepstral_, num_cepstral_);

  WriteToken(out_stream, binary, "<OCCUPANCY>");
  occupancy_bf.CopyFromVec(occupancy_);
  occupancy_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MUVS>");
  vec_bf.CopyFromMat(mu_vs_);
  vec_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MUVD>");
  vec_bf.CopyFromMat(mu_vd_);
  vec_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MUVA>");
  vec_bf.CopyFromMat(mu_va_);
  vec_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MUMS>");
  mat_bf.CopyFromMat(mu_ms_);
  mat_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MUMD>");
  mat_bf.CopyFromMat(mu_md_);
  mat_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<MUMA>");
  mat_bf.CopyFromMat(mu_ma_);
  mat_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<VARJS>");
  vec_bf.CopyFromMat(var_js_);
  vec_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<VARJD>");
  vec_bf.CopyFromMat(var_jd_);
  vec_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<VARJA>");
  vec_bf.CopyFromMat(var_ja_);
  vec_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<VARHS>");
  mat_bf.CopyFromMat(var_hs_);
  mat_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<VARHD>");
  mat_bf.CopyFromMat(var_hd_);
  mat_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<VARHA>");
  mat_bf.CopyFromMat(var_ha_);
  mat_bf.Write(out_stream, binary);
  WriteToken(out_stream, binary, "</VTSGMMACCS>");

}

void VtsAccumDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags) {
    KALDI_ERR<< "Flags in argument do not match the active accumulators";
  }
  if (flags & kGmmWeights) occupancy_.SetZero();
  if (flags & kGmmMeans) {
    mu_vs_.SetZero();
    mu_vd_.SetZero();
    mu_va_.SetZero();
    mu_ms_.SetZero();
    mu_md_.SetZero();
    mu_ma_.SetZero();
  }
  if (flags & kGmmVariances) {
    var_js_.SetZero();
    var_jd_.SetZero();
    var_ja_.SetZero();
    var_hs_.SetZero();
    var_hd_.SetZero();
    var_ha_.SetZero();
  }
}

void VtsAccumDiagGmm::Scale(BaseFloat f, GmmFlagsType flags) {
  if (flags & ~flags) {
    KALDI_ERR<< "Flags in argument do not match the active accumulators";
  }
  double d = static_cast<double>(f);
  if (flags & kGmmWeights) occupancy_.Scale(d);
  if (flags & kGmmMeans) {
    mu_vs_.Scale(d);
    mu_vd_.Scale(d);
    mu_va_.Scale(d);
    mu_ms_.Scale(d);
    mu_md_.Scale(d);
    mu_ma_.Scale(d);
  }
  if (flags & kGmmVariances) {
    var_js_.Scale(d);
    var_jd_.Scale(d);
    var_ja_.Scale(d);
    var_hs_.Scale(d);
    var_hd_.Scale(d);
    var_ha_.Scale(d);
  }
}

void VtsAccumDiagGmm::Add(double scale, const VtsAccumDiagGmm &acc) {
  // In C++, object can access the private members of another object
  // of the same class, i.e. each class is implicitly the friend of itself.
  // The function will crash if the dimensions are not matching.
  occupancy_.AddVec(scale, acc.occupancy_);
  if (flags_ & kGmmMeans) {
    mu_vs_.AddMat(scale, acc.mu_vs_);
    mu_vd_.AddMat(scale, acc.mu_vd_);
    mu_va_.AddMat(scale, acc.mu_va_);
    mu_ms_.AddMat(scale, acc.mu_ms_);
    mu_md_.AddMat(scale, acc.mu_md_);
    mu_ma_.AddMat(scale, acc.mu_ma_);
  }

  if (flags_ & kGmmVariances) {
    var_js_.AddMat(scale, acc.var_js_);
    var_jd_.AddMat(scale, acc.var_jd_);
    var_ja_.AddMat(scale, acc.var_ja_);
    var_hs_.AddMat(scale, acc.var_hs_);
    var_hd_.AddMat(scale, acc.var_hd_);
    var_ha_.AddMat(scale, acc.var_ha_);
  }
}

BaseFloat VtsAccumDiagGmm::AccumulateFromDiag(
    const DiagGmm &clean_gmm,
    const DiagGmm &noisy_gmm,
    const std::vector<Matrix<double> > &Jx,
    int32 offset,
    const VectorBase<BaseFloat> &data,
    BaseFloat frame_posterior) {
  KALDI_ASSERT(noisy_gmm.NumGauss() == NumGauss());
  KALDI_ASSERT(noisy_gmm.Dim() == Dim());
  KALDI_ASSERT(static_cast<int32>(data.Dim()) == Dim());

  Vector<BaseFloat> posteriors(NumGauss());
  BaseFloat log_like = noisy_gmm.ComponentPosteriors(data, &posteriors);
  posteriors.Scale(frame_posterior);

  // convert to double
  Vector<double> post_d(posteriors);

  // accumulate occupancy
  occupancy_.AddVec(1.0, post_d);

  /*
   * Starting accumulate the necessary statistics for mean update
   */
  if (flags_ & kGmmMeans) {
    int32 num_gauss = NumGauss();

    // for mean update only noise model is needed
    DiagGmmNormal ngmm_noisy(noisy_gmm);

    // iterate through each Gaussian
    for (int32 g = 0; g < num_gauss; ++g) {
      // inverse variance
      Vector<double> inv_var(ngmm_noisy.vars_.Row(g));
      inv_var.InvertElements();

      Matrix<double> Jx_sm(Jx[offset + g]);

      Vector<double> y_mu(data);
      y_mu.AddVec(-1.0, ngmm_noisy.means_.Row(g));  // y - mu

      Matrix<double> tmp_mat(num_cepstral_, num_cepstral_);
      // static
      tmp_mat.CopyFromMat(Jx_sm, kTrans);  // Jx^T
      tmp_mat.MulColsVec(SubVector<double>(inv_var, 0, num_cepstral_));  // Jx^T * inv_var
      SubMatrix<double> cur_mu_ms(mu_ms_, g * num_cepstral_, num_cepstral_, 0,
                                  num_cepstral_);
      cur_mu_ms.AddMatMat(post_d(g), tmp_mat, kNoTrans, Jx_sm, kNoTrans, 1.0);  // gamma * Jx^T * inv_var * Jx
      mu_vs_.Row(g).AddMatVec(post_d(g), tmp_mat, kNoTrans,
                              SubVector<double>(y_mu, 0, num_cepstral_), 1.0);  // gamma * Jx^T * inv_var * (y - mu)
      // delta
      tmp_mat.CopyFromMat(Jx_sm, kTrans);
      tmp_mat.MulColsVec(
          SubVector<double>(inv_var, num_cepstral_, num_cepstral_));
      SubMatrix<double> cur_mu_md(mu_md_, g * num_cepstral_, num_cepstral_, 0,
                                  num_cepstral_);
      cur_mu_md.AddMatMat(post_d(g), tmp_mat, kNoTrans, Jx_sm, kNoTrans, 1.0);
      mu_vd_.Row(g).AddMatVec(
          post_d(g), tmp_mat, kNoTrans,
          SubVector<double>(y_mu, num_cepstral_, num_cepstral_), 1.0);
      // accelerate
      tmp_mat.CopyFromMat(Jx_sm, kTrans);
      tmp_mat.MulColsVec(
          SubVector<double>(inv_var, num_cepstral_ << 1, num_cepstral_));
      SubMatrix<double> cur_mu_ma(mu_ma_, g * num_cepstral_, num_cepstral_, 0,
                                  num_cepstral_);
      cur_mu_ma.AddMatMat(post_d(g), tmp_mat, kNoTrans, Jx_sm, kNoTrans, 1.0);
      mu_va_.Row(g).AddMatVec(
          post_d(g), tmp_mat, kNoTrans,
          SubVector<double>(y_mu, num_cepstral_ << 1, num_cepstral_), 1.0);

    }

    /*
     * Starting accumulate statistics for variance
     * don't allow update variance only
     */
    if (flags_ & kGmmVariances) {

      // variance stats needs the clean model variance
      DiagGmmNormal ngmm_clean(clean_gmm);

      // iterate through each Gaussian
      for (int32 g = 0; g < num_gauss; ++g) {
        // inverse variance
        Vector<double> inv_var(ngmm_noisy.vars_.Row(g));
        inv_var.InvertElements();

        Matrix<double> Jx_sm2(Jx[offset + g]);
        Jx_sm2.ApplyPow(2.0);  // Jx^2

        Vector<double> y_mu_inv2(data);  // y
        y_mu_inv2.AddVec(-1.0, ngmm_noisy.means_.Row(g));  // y - mu
        y_mu_inv2.MulElements(inv_var);  // (y - mu)/var
        y_mu_inv2.ApplyPow(2.0);  // ((y - mu)/var)^2

        Vector<double> tmp_vec(num_cepstral_), cur_jac(num_cepstral_);
        Matrix<double> tmp_mat(num_cepstral_, num_cepstral_), tmp_mat2(
            num_cepstral_, num_cepstral_);

        // static - Jacobian
        tmp_vec.CopyFromVec(SubVector<double>(inv_var, 0, num_cepstral_));
        tmp_vec.AddVec(-1.0, SubVector<double>(y_mu_inv2, 0, num_cepstral_));
        cur_jac.AddMatVec(1.0, Jx_sm2, kTrans, tmp_vec, 0.0);
        cur_jac.MulElements(
            SubVector<double>(ngmm_clean.vars_.Row(g), 0, num_cepstral_));
        var_js_.Row(g).AddVec(post_d(g), cur_jac);
        // static - Hessian
        tmp_vec.SetZero();
        tmp_vec.AddVec(-1.0, SubVector<double>(inv_var, 0, num_cepstral_));
        tmp_vec.AddVec(2.0, SubVector<double>(y_mu_inv2, 0, num_cepstral_));
        tmp_vec.MulElements(SubVector<double>(inv_var, 0, num_cepstral_));  //
        tmp_mat.CopyFromMat(Jx_sm2);  //
        tmp_mat.MulRowsVec(tmp_vec);  // diag(tmp_vec) * tmp_mat
        tmp_mat2.AddMatMat(1.0, Jx_sm2, kTrans, tmp_mat, kNoTrans, 0.0);
        tmp_mat.SetZero();
        tmp_mat.AddVecVec(
            1.0, SubVector<double>(ngmm_clean.vars_.Row(g), 0, num_cepstral_),
            SubVector<double>(ngmm_clean.vars_.Row(g), 0, num_cepstral_));
        tmp_mat.MulElements(tmp_mat2);
        tmp_mat2.SetZero();
        tmp_mat2.CopyDiagFromVec(cur_jac);  // deal with the extra term for diagonal
        tmp_mat.AddMat(1.0, tmp_mat2, kNoTrans);
        SubMatrix<double> cur_var_hs(var_hs_, g * num_cepstral_, num_cepstral_,
                                     0, num_cepstral_);
        cur_var_hs.AddMat(post_d(g), tmp_mat);

        // delta - Jacobian
        tmp_vec.CopyFromVec(
            SubVector<double>(inv_var, num_cepstral_, num_cepstral_));
        tmp_vec.AddVec(
            -1.0, SubVector<double>(y_mu_inv2, num_cepstral_, num_cepstral_));
        cur_jac.AddMatVec(1.0, Jx_sm2, kTrans, tmp_vec, 0.0);
        cur_jac.MulElements(
            SubVector<double>(ngmm_clean.vars_.Row(g), num_cepstral_,
                              num_cepstral_));
        var_jd_.Row(g).AddVec(post_d(g), cur_jac);
        // delta - Hessian
        tmp_vec.SetZero();
        tmp_vec.AddVec(
            -1.0, SubVector<double>(inv_var, num_cepstral_, num_cepstral_));
        tmp_vec.AddVec(
            2.0, SubVector<double>(y_mu_inv2, num_cepstral_, num_cepstral_));
        tmp_vec.MulElements(
            SubVector<double>(inv_var, num_cepstral_, num_cepstral_));  //
        tmp_mat.CopyFromMat(Jx_sm2);  //
        tmp_mat.MulRowsVec(tmp_vec);  // diag(tmp_vec) * tmp_mat
        tmp_mat2.AddMatMat(1.0, Jx_sm2, kTrans, tmp_mat, kNoTrans, 0.0);
        tmp_mat.SetZero();
        tmp_mat.AddVecVec(
            1.0,
            SubVector<double>(ngmm_clean.vars_.Row(g), num_cepstral_,
                              num_cepstral_),
            SubVector<double>(ngmm_clean.vars_.Row(g), num_cepstral_,
                              num_cepstral_));
        tmp_mat.MulElements(tmp_mat2);
        tmp_mat2.SetZero();
        tmp_mat2.CopyDiagFromVec(cur_jac);  // deal with the extra term for diagonal
        tmp_mat.AddMat(1.0, tmp_mat2, kNoTrans);
        SubMatrix<double> cur_var_hd(var_hd_, g * num_cepstral_, num_cepstral_,
                                     0, num_cepstral_);
        cur_var_hd.AddMat(post_d(g), tmp_mat);

        // accelerate - Jacobian
        tmp_vec.CopyFromVec(
            SubVector<double>(inv_var, num_cepstral_ << 1, num_cepstral_));
        tmp_vec.AddVec(
            -1.0,
            SubVector<double>(y_mu_inv2, num_cepstral_ << 1, num_cepstral_));
        cur_jac.AddMatVec(1.0, Jx_sm2, kTrans, tmp_vec, 0.0);
        cur_jac.MulElements(
            SubVector<double>(ngmm_clean.vars_.Row(g), num_cepstral_ << 1,
                              num_cepstral_));
        var_ja_.Row(g).AddVec(post_d(g), cur_jac);
        // accelerate - Hessian
        tmp_vec.SetZero();
        tmp_vec.AddVec(
            -1.0,
            SubVector<double>(inv_var, num_cepstral_ << 1, num_cepstral_));
        tmp_vec.AddVec(
            2.0,
            SubVector<double>(y_mu_inv2, num_cepstral_ << 1, num_cepstral_));
        tmp_vec.MulElements(
            SubVector<double>(inv_var, num_cepstral_ << 1, num_cepstral_));  //
        tmp_mat.CopyFromMat(Jx_sm2);  //
        tmp_mat.MulRowsVec(tmp_vec);  // diag(tmp_vec) * tmp_mat
        tmp_mat2.AddMatMat(1.0, Jx_sm2, kTrans, tmp_mat, kNoTrans, 0.0);
        tmp_mat.SetZero();
        tmp_mat.AddVecVec(
            1.0,
            SubVector<double>(ngmm_clean.vars_.Row(g), num_cepstral_ << 1,
                              num_cepstral_),
            SubVector<double>(ngmm_clean.vars_.Row(g), num_cepstral_ << 1,
                              num_cepstral_));
        tmp_mat.MulElements(tmp_mat2);
        tmp_mat2.SetZero();
        tmp_mat2.CopyDiagFromVec(cur_jac);  // deal with the extra term for diagonal
        tmp_mat.AddMat(1.0, tmp_mat2, kNoTrans);
        SubMatrix<double> cur_var_ha(var_ha_, g * num_cepstral_, num_cepstral_,
                                     0, num_cepstral_);
        cur_var_ha.AddMat(post_d(g), tmp_mat);

      }

    }
  }

  return log_like;

}

void VtsDiagGmmUpdate(const VtsDiagGmmOptions &config,
                      const VtsAccumDiagGmm &diaggmm_acc,
                      GmmFlagsType flags,
                      DiagGmm *gmm,
                      BaseFloat *obj_change_out,
                      BaseFloat *count_out) {
  KALDI_ASSERT(gmm!=NULL);

  if (flags & ~diaggmm_acc.Flags()) {
    KALDI_ERR<< "Flags in argument do not match the active accumulators";
  }

  int32 num_gauss = gmm->NumGauss();
  double occ_sum = diaggmm_acc.occupancy().Sum();

  int32 tot_floored = 0, gauss_floored = 0;

  // remember old objective value
  gmm->ComputeGconsts();
  BaseFloat obj_old = 0.0;  //TODO:: compute objective value

  // allocate the gmm in normal representation; all parameters of this will be
  // updated, but only the flagged ones will be transfered back to gmm
  DiagGmmNormal ngmm(*gmm);
  Vector<double> mean_update(gmm->Dim());
  Vector<double> var_update(gmm->Dim());

  int32 num_cepstral = diaggmm_acc.NumCepstral();
  Vector<double> tmp_vec(num_cepstral);
  Matrix<double> tmp_mat(num_cepstral, num_cepstral);

  std::vector<int32> to_remove;
  for (int32 i = 0; i < num_gauss; ++i) {
    double occ = diaggmm_acc.occupancy()(i);
    double prob;
    if (occ_sum > 0.0) {
      prob = occ / occ_sum;
    } else {
      prob = 1.0 / num_gauss;
    }

    if (occ > static_cast<double>(config.min_gaussian_occupancy)
        && prob > static_cast<double>(config.min_gaussian_weight)) {
      ngmm.weights_(i) = prob;

      // update mean, then variances
      if (diaggmm_acc.Flags() & kGmmMeans) {
        // static
        SubVector<double> s_update(mean_update, 0, num_cepstral);
        tmp_mat.CopyFromMat(
            SubMatrix<double>(diaggmm_acc.mu_ms(), i * num_cepstral,
                              num_cepstral,
                              0, num_cepstral));
        tmp_mat.Invert();
        s_update.AddMatVec(1.0, tmp_mat, kNoTrans, diaggmm_acc.mu_vs().Row(i),
                           0.0);
        // delta
        SubVector<double> d_update(mean_update, num_cepstral, num_cepstral);
        tmp_mat.CopyFromMat(
            SubMatrix<double>(diaggmm_acc.mu_md(), i * num_cepstral,
                              num_cepstral,
                              0, num_cepstral));
        tmp_mat.Invert();
        d_update.AddMatVec(1.0, tmp_mat, kNoTrans, diaggmm_acc.mu_vd().Row(i),
                           0.0);
        // accelerate
        SubVector<double> a_update(mean_update, num_cepstral << 1,
                                   num_cepstral);
        tmp_mat.CopyFromMat(
            SubMatrix<double>(diaggmm_acc.mu_ma(), i * num_cepstral,
                              num_cepstral,
                              0, num_cepstral));
        tmp_mat.Invert();
        a_update.AddMatVec(1.0, tmp_mat, kNoTrans, diaggmm_acc.mu_va().Row(i),
                           0.0);

        // transfer to update
        (ngmm.means_.Row(i)).AddVec(1.0, mean_update);
      }

      if (diaggmm_acc.Flags() & kGmmVariances) {
        // disallow updating variance alone
        KALDI_ASSERT(diaggmm_acc.Flags() & kGmmMeans);
        // original variance
        Vector<double> var(ngmm.vars_.Row(i));

        // diagonal loading matrix
        Matrix<double> dload(num_cepstral, num_cepstral);
        dload.SetUnit();
        dload.Scale(config.diagonal_loading);

        // static
        SubVector<double> s_update(var_update, 0, num_cepstral);
        tmp_mat.CopyFromMat(
            SubMatrix<double>(diaggmm_acc.var_hs(), i * num_cepstral,
                              num_cepstral,
                              0, num_cepstral));
        tmp_mat.AddMat(-1.0, dload);
        tmp_mat.Invert();
        s_update.AddMatVec(config.variance_lrate, tmp_mat, kNoTrans,
                           diaggmm_acc.var_js().Row(i),
                           0.0);
        // delta
        SubVector<double> d_update(var_update, num_cepstral, num_cepstral);
        tmp_mat.CopyFromMat(
            SubMatrix<double>(diaggmm_acc.var_hd(), i * num_cepstral,
                              num_cepstral,
                              0, num_cepstral));
        tmp_mat.AddMat(-1.0, dload);
        tmp_mat.Invert();
        d_update.AddMatVec(config.variance_lrate, tmp_mat, kNoTrans,
                           diaggmm_acc.var_jd().Row(i),
                           0.0);
        // accelerate
        SubVector<double> a_update(var_update, num_cepstral << 1, num_cepstral);
        tmp_mat.CopyFromMat(
            SubMatrix<double>(diaggmm_acc.var_ha(), i * num_cepstral,
                              num_cepstral,
                              0, num_cepstral));
        tmp_mat.AddMat(-1.0, dload);
        tmp_mat.Invert();
        a_update.AddMatVec(config.variance_lrate, tmp_mat, kNoTrans,
                           diaggmm_acc.var_ja().Row(i),
                           0.0);

        // Limit the variance change
        for (int32 k = 0; k < var_update.Dim(); ++k) {
          if (var_update(k) > config.stigma) {
            var_update(k) = config.stigma;
          }
          if (var_update(k) < -config.stigma) {
            var_update(k) = -config.stigma;
          }
        }
        var_update.ApplyExp();
        var_update.InvertElements();
        var.MulElements(var_update);

        int32 floored;
        if (config.variance_floor_vector.Dim() != 0) {
          floored = var.ApplyFloor(config.variance_floor_vector);
        } else {
          floored = var.ApplyFloor(config.min_variance);
        }

        if (floored != 0) {
          tot_floored += floored;
          gauss_floored++;
        }

        // transfer to estimate
        ngmm.vars_.CopyRowFromVec(var, i);
      }

    } else {  // Insufficient occupancy
      if (config.remove_low_count_gaussians &&
          static_cast<int32>(to_remove.size()) < num_gauss - 1) {
        // remove the component, unless it is the last one
        KALDI_WARN<< "Too little data - removing Gaussian (weight "
        << std::fixed << prob
        << ", occupation count " << std::fixed << diaggmm_acc.occupancy()(i)
        << ", vector size " << gmm->Dim() << ")";
        to_remove.push_back(i);
      } else {
        KALDI_WARN << "Gaussian has too little data but not removing it because"
        << (config.remove_low_count_gaussians ?
            " it is the last Gaussian: i = "
            : " remove-low-count-gaussians == false: g = ") << i
        << ", occ = " << diaggmm_acc.occupancy()(i) << ", weight = " << prob;
        ngmm.weights_(i) =
        std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }

        // copy to natural representation according to flags
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();
  BaseFloat obj_new = 0.0;  //TODO:: compute objective

  if (obj_change_out) {
    KALDI_WARN<< "Objective change has not been implemented yet, which will always be 0.";
    *obj_change_out = (obj_new - obj_old);
  }

  if (count_out) {
    *count_out = occ_sum;
  }

  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /*renormalize weights */);
    gmm->ComputeGconsts();
  }

  if (tot_floored > 0) {
    KALDI_WARN<< tot_floored << " variances floored in " << gauss_floored
    << " Gaussians.";
  }
}

}  // End namespace kaldi
