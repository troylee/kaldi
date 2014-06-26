/*
 * nnet-cmvnbl.h
 *
 *  Created on: Jan 15, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  CMVN Normalization Layer
 *
 *  Currently only support the FBank input features.
 *
 *  This layer is mainly used for CMVN and noise parameter
 *  estimation, better not to use in saved nnet models.
 *
 *  Feature must be FBANK_D_A with/without _E and _E is the last element.
 *
 *
 */

#ifndef KALDI_NNET_CMVNBL_H
#define KALDI_NNET_CMVNBL_H

#include "nnet/nnet-component.h"

namespace kaldi {

// CMVN BiasedLinearity
class CMVNBL : public UpdatableComponent {
 public:
  CMVNBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : UpdatableComponent(dim_in, dim_out, nnet),
        feat_dim_(dim_in),
        win_len_(dim_out)
  {
    // Initialize the weights to be identity and bias to 0
    MatrixIndexT dim = feat_dim_ * win_len_;

    input_dim_ = dim;  // real dimensions
    output_dim_ = dim;

    linearity_.Resize(dim, dim);
    bias_.Resize(dim);
    bias_.SetZero();
    host_bias_.Resize(dim);
    host_linearity_.Resize(dim, dim, kSetZero);
    for (MatrixIndexT i = 0; i < dim; ++i) {
      host_linearity_(i, i) = 1.0;
    }
    linearity_.CopyFromMat(host_linearity_);

    // Initial normalization parameters
    clean_mu_.Resize(feat_dim_, kSetZero);
    clean_var_.Resize(feat_dim_, kSetZero);
    clean_var_.Set(1.0);

    noise_mu_.Resize(feat_dim_, kSetZero);
    noise_var_.Resize(feat_dim_, kSetZero);

    linearity_corr_.Resize(dim, dim);
    bias_corr_.Resize(dim);

    have_noise_ = false;

    // default parameter kind
    have_energy_ = true;
    num_fbank_ = 40;
    delta_order_ = 3;

    update_flag_ = "";
    update_var_ = false;

    stigma_ = 1.0;

  }
  ~CMVNBL()
  {
  }

  ComponentType GetType() const {
    return kCMVNBL;
  }

  void ReadData(std::istream &is, bool binary) {
    // no data to read
  }

  void WriteData(std::ostream &os, bool binary) const {
    // no data to write
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

    if (update_flag_ != "cmvn" && update_flag_ != "noise") {
      return;  // nothing to update except these two
    }

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
    //linearity_corr_.AddMatMat(1.0, err, kTrans, input, kNoTrans, momentum_);
    //bias_corr_.AddRowSumMat(1.0, err, momentum_);
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
     */
    // update
    // In the CMVN layer, this function is only used to compute the weight changes
    //linearity_.AddMat(-learn_rate_, linearity_corr_);
    //bias_.AddVec(-learn_rate_, bias_corr_);
    if (update_flag_ == "cmvn") {
      UpdateCMVN(input.NumRows());
    } else if (update_flag_ == "noise") {
      UpdateNoise(input.NumRows());
    }

    // update the layer weights
    UpdateLayerWeights();

  }

  void SetUpdateFlag(const std::string &flag, bool update_var) {
    update_flag_ = flag;
    update_var_ = update_var;
  }

  void SetParamKind(bool have_energy, int32 num_fbank, int32 delta_order) {
    have_energy_ = have_energy;
    num_fbank_ = num_fbank;

    if (delta_order < 1 || delta_order > 3) {
      KALDI_ERR<< "Only delta order of 1,2,3 are supported";
    }

    delta_order_ = delta_order;

    if (num_fbank_ != (feat_dim_ / delta_order_ - (have_energy ? 1 : 0))) {
      KALDI_ERR<< "Inconsistant feature configuration: [have_energy="
      << have_energy << ", feat_dim="<< feat_dim_
      << ", num_fbank="<< num_fbank_<< ", delta_order="
      << delta_order_ <<"]";
    }

    mu_h_.Resize(num_fbank_ * delta_order_, kSetZero);
    mu_z_.Resize(num_fbank_ * delta_order_, kSetZero);
    var_z_.Resize(num_fbank_ * delta_order_, kSetZero);
    vec_Jx_.Resize(num_fbank_, kSetZero);
    vec_Jz_.Resize(num_fbank_, kSetZero);
  }

  // return the mean and variance of the normalization
  void GetCMVN(Vector<double> &mu, Vector<double> &var) {
    mu.CopyFromVec(clean_mu_);
    var.CopyFromVec(clean_var_);
  }

  // Set mean and variance of the normalization
  void SetCMVN(const Vector<double> &mu, const Vector<double> &var) {
    clean_mu_.CopyFromVec(mu);
    clean_var_.CopyFromVec(var);

    UpdateLayerWeights();
  }

  void GetNoise(Vector<double> &mu_h, Vector<double> &mu_z,
                Vector<double> &var_z) {
    mu_h.CopyFromVec(mu_h_);
    mu_z.CopyFromVec(mu_z_);
    var_z.CopyFromVec(var_z_);
  }

  void SetNoise(const Vector<double> &mu_h, const Vector<double> &mu_z,
                const Vector<double> &var_z) {
    mu_h_.CopyFromVec(mu_h);
    mu_z_.CopyFromVec(mu_z);
    var_z_.CopyFromVec(var_z);

    have_noise_ = true;

    UpdateLayerWeights();
  }

  void ClearNoise() {
    have_noise_ = false;

    UpdateLayerWeights();
  }

 private:

  void UpdateCMVN(int32 num) {
    /*
     * Update mean
     */
    // compute gradient
    Vector<BaseFloat> bias_corr(bias_corr_.Dim(), kSetZero);
    bias_corr_.CopyToVec(&bias_corr);
    Vector<double> mu_corr(feat_dim_, kSetZero);
    // sum different frames together
    for (int32 f = 0; f < win_len_; ++f) {
      for (int32 i = 0; i < feat_dim_; ++i) {
        mu_corr(i) += static_cast<double>(bias_corr(f * feat_dim_ + i));
      }
    }
    Vector<double> inv_var(noise_var_);  // var_g
    inv_var.ApplyPow(-0.5);  // 1.0 / sqrt(var_g)
    inv_var.Scale(-1.0);  // -1.0 / sqrt(var_g)
    for (int32 d = 0, k = 0, j = 0; d < delta_order_; ++d) {
      for (int32 i = 0; i < num_fbank_; ++i, ++j, ++k) {
        mu_corr(k) *= (inv_var(j) * vec_Jx_(i));  // d_bias * (-1.0 / sqrt(var_g)) * Jx
      }

      if (have_energy_) {  // advance the correction if have energy
        ++k;
      }
    }

    // update the mean
    clean_mu_.AddVec(-learn_rate_ / (win_len_ * num), mu_corr);

    if (update_var_) {
      /*
       * Update variance
       */
      Matrix<BaseFloat> linearity_corr(linearity_corr_.NumRows(),
                                       linearity_corr_.NumCols(),
                                       kSetZero);
      linearity_corr_.CopyToMat(&linearity_corr);
      Vector<double> var_corr(feat_dim_, kSetZero);
      // sum over different frames together
      for (int32 f = 0, k = 0; f < win_len_; ++f) {
        for (int32 i = 0; i < feat_dim_; ++i, ++k) {
          var_corr(i) += static_cast<double>(-0.5 * linearity_corr(k, k)
              + 0.5 * bias_corr(k) * noise_mu_(i));
        }
      }

      inv_var.Scale(-1.0);  // 1.0 / sqrt(var_g)
      inv_var.ApplyPow(3.0);  // 1.0 / sqrt(var_g)^3
      for (int32 d = 0, k = 0, j = 0; d < delta_order_; ++d) {
        for (int32 i = 0; i < num_fbank_; ++i, ++j, ++k) {
          var_corr(j) *= (inv_var(k) * vec_Jx_(i) * vec_Jx_(i) * clean_var_(j));
        }

        if (have_energy_) {  // advance the correction if have energy
          ++k;
        }
      }

      // update the noise
      var_corr.Scale(-learn_rate_ / (win_len_ * num));
      for (int32 i = 0; i < feat_dim_; ++i) {
        if (var_corr(i) > stigma_) {
          var_corr(i) = stigma_;
        }
        if (var_corr(i) < -stigma_) {
          var_corr(i) = -stigma_;
        }
      }
      var_corr.ApplyExp();
      clean_var_.MulElements(var_corr);
    }
  }

  void UpdateNoise(int32 num) {
    /*
     * Update mean
     *
     */
    // compute graident
    Vector<BaseFloat> bias_corr(bias_corr_.Dim(), kSetZero);
    bias_corr_.CopyToVec(&bias_corr);
    Vector<double> mu_h_corr(num_fbank_ * delta_order_, kSetZero);
    Vector<double> mu_z_corr(num_fbank_ * delta_order_, kSetZero);
    // sum over different frames together
    for (int32 f = 0; f < win_len_; ++f) {
      for (int32 i = 0; i < num_fbank_; ++i) {  // only the static part
        mu_h_corr(i) += static_cast<double>(bias_corr(f * feat_dim_ + i));
      }
    }
    mu_z_corr.CopyFromVec(mu_h_corr);
    Vector<double> inv_var(noise_var_);
    inv_var.ApplyPow(-0.5);
    inv_var.Scale(-1.0);  // - 1.0 / sqrt(var_g)
    for (int32 i = 0; i < num_fbank_; ++i) {
      mu_h_corr(i) *= (inv_var(i) * vec_Jx_(i));
      mu_z_corr(i) *= (inv_var(i) * vec_Jz_(i));
    }
    // update the noise
    mu_h_.AddVec(-learn_rate_, mu_h_corr);
    mu_z_.AddVec(-learn_rate_, mu_z_corr);

    if (update_var_) {
      /*
       * Update variance
       */
      Matrix<BaseFloat> linearity_corr(linearity_corr_.NumRows(),
                                       linearity_corr_.NumCols(),
                                       kSetZero);
      linearity_corr_.CopyToMat(&linearity_corr);

      Vector<double> var_z_corr(num_fbank_ * delta_order_, kSetZero);
      // sum over different frames together
      for (int32 f = 0, t = 0; f < win_len_; ++f) {
        for (int32 d = 0, k = 0, j = 0; d < delta_order_; ++d) {
          for (int32 i = 0; i < num_fbank_; ++i, ++j, ++k, ++t) {
            var_z_corr(j) += static_cast<double>(-0.5 * linearity_corr(t, t)
                + 0.5 * bias_corr(t) * noise_mu_(k));
          }
          if (have_energy_) {
            ++k;  // bypass the energy component if necessary
          }
        }
      }
      inv_var.Scale(-1.0);  // 1.0 / sqrt(var_g)
      inv_var.ApplyPow(3.0);  // 1.0 / sqrt(var_g)^3
      for (int32 d = 0, k = 0, j = 0; d < delta_order_; ++d) {
        for (int32 i = 0; i < num_fbank_; ++i, ++j, ++k) {
          var_z_corr(j) *= (inv_var(k) * vec_Jz_(i) * vec_Jz_(i) * var_z_(j));
        }
        if (have_energy_) {
          ++k;
        }
      }

      // update the noise
      var_z_corr.Scale(-learn_rate_);
      for (int32 i = 0; i < num_fbank_ * delta_order_; ++i) {
        if (var_z_corr(i) > stigma_) {
          var_z_corr(i) = stigma_;
        }
        if (var_z_corr(i) < -stigma_) {
          var_z_corr(i) = -stigma_;
        }
      }
      var_z_corr.ApplyExp();
      var_z_.MulElements(var_z_corr);
    }
  }

  void UpdateLayerWeights() {
    noise_mu_.CopyFromVec(clean_mu_);
    noise_var_.CopyFromVec(clean_var_);

    if (have_noise_) {
      // compensate the mean and variance
      Matrix<double> Jx, Jz;
      CompensateDiagGaussian_FBank(mu_h_, mu_z_, var_z_, have_energy_,
                                   num_fbank_,
                                   delta_order_,
                                   noise_mu_,
                                   noise_var_,
                                   Jx,
                                   Jz);
      vec_Jx_.CopyDiagFromMat(Jx);
      vec_Jz_.CopyDiagFromMat(Jz);

    }

    Vector<double> inv_std(feat_dim_, kSetZero);
    for (int32 j = 0; j < feat_dim_; ++j) {
      inv_std(j) = 1.0 / sqrt(noise_var_(j));
    }

    for (int32 i = 0; i < win_len_; ++i) {
      for (int32 j = 0; j < feat_dim_; ++j) {
        host_linearity_(i * feat_dim_ + j, i * feat_dim_ + j) =
            static_cast<BaseFloat>(inv_std(j));
        host_bias_(i * feat_dim_ + j) = static_cast<BaseFloat>(-noise_mu_(j)
            * inv_std(j));
      }
    }
    linearity_.CopyFromMat(host_linearity_);
    bias_.CopyFromVec(host_bias_);
  }

  void CompensateDiagGaussian_FBank(const Vector<double> &mu_h,
                                    const Vector<double> &mu_z,
                                    const Vector<double> &var_z,
                                    bool have_energy,
                                    int32 num_fbank,
                                    int32 delta_order,
                                    Vector<double> &mean,
                                    Vector<double> &cov,
                                    Matrix<double> &Jx,
                                    Matrix<double> &Jz) {
    KALDI_ASSERT(delta_order>=1 && delta_order<=3);

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
    for (int32 ii = 0; ii < num_fbank; ++ii)
      Jz(ii, ii) = 1.0 - Jz(ii, ii);

    // compute and update mean
    if (g_kaldi_verbose_level >= 9) {
      KALDI_LOG<< "Mean Before: " << mean;
    }
    Vector<double> tmp_mu(num_fbank);
    SubVector<double> mu_s(mean, 0, num_fbank);
    mu_s.CopyFromVec(mu_y_s);
    if (delta_order >= 2) {
      SubVector<double> mu_dt(mean, num_fbank + (have_energy ? 1 : 0),
                              num_fbank);
      tmp_mu.CopyFromVec(mu_dt);
      mu_dt.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
    }
    if (delta_order == 3) {
      SubVector<double> mu_acc(mean, 2 * (num_fbank + (have_energy ? 1 : 0)),
                               num_fbank);
      tmp_mu.CopyFromVec(mu_acc);
      mu_acc.AddMatVec(1.0, Jx, kNoTrans, tmp_mu, 0.0);
    }
    if (g_kaldi_verbose_level >= 9) {
      KALDI_LOG<< "Mean After: " << mean;
    }

      // compute and update covariance
    if (g_kaldi_verbose_level >= 9) {
      KALDI_LOG<< "Covarianc Before: " << cov;
    }
    for (int32 ii = 0; ii < delta_order; ++ii) {
      Matrix<double> tmp_var1(Jx), tmp_var2(Jz), new_var(num_fbank,
                                                         num_fbank);
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
      KALDI_LOG<< "Covariance After: " << cov;
    }
  }

private:
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  int32 feat_dim_;
  int32 win_len_;

  Matrix<BaseFloat> host_linearity_;
  Vector<BaseFloat> host_bias_;

// normalization parameters
  Vector<double> clean_mu_;
  Vector<double> clean_var_;

// noise parameters
  bool have_noise_;
  Vector<double> mu_h_;
  Vector<double> mu_z_;
  Vector<double> var_z_;

// noise compensated normalization params
  Vector<double> noise_mu_;
  Vector<double> noise_var_;

// Jacobians, as they are diagonal for FBanks, we use vectors
  Vector<double> vec_Jx_;
  Vector<double> vec_Jz_;

// whether the feature has energy
  bool have_energy_;
  int32 num_fbank_;
  int32 delta_order_;

// which parameter to update, 'cmvn' or 'noise'
  std::string update_flag_;
// whether variance is updated
  bool update_var_;

// variance update limit
  double stigma_;
}
;

}  // namespace

#endif
