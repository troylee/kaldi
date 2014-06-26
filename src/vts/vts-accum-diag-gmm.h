/*
 * vts/vts-accum-diag-gmm.h
 *
 *  Created on: Nov 23, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef VTS_VTS_ACCUM_DIAG_GMM_H_
#define VTS_VTS_ACCUM_DIAG_GMM_H_

#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/model-common.h"
#include "util/parse-options.h"

namespace kaldi {

/** \struct VtsDiagGmmOptions
 *  Configuration variables like variance floor, minimum occupancy, etc.
 *  needed in the estimation process.
 */
struct VtsDiagGmmOptions {
  /// Flags to control which parameters to update

  /// Variance floor for each dimension [empty if not supplied].
  /// It is in double since the variance is computed in double precision.
  Vector<double> variance_floor_vector;
  /// Minimum weight below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true);
  BaseFloat min_gaussian_weight;
  /// Minimum count below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true).
  BaseFloat min_gaussian_occupancy;
  /// Minimum allowed variance in any dimension (if no variance floor)
  /// It is in double since the variance is computed in double precision.
  double min_variance;
  bool remove_low_count_gaussians;
  /// Diagonal loading to ensure the Hessian matrix to be negative definite
  /// Refer to Ozlem Kalinli's Noise Adaptive Training for robust ASR
  double diagonal_loading;
  /// Limit the change amount of the log(variance)
  double stigma;
  /// Learning rate for the variance
  double variance_lrate;
  VtsDiagGmmOptions() {
    // don't set var floor vector by default.
    min_gaussian_weight     = 1.0e-05;
    min_gaussian_occupancy  = 10.0;
    min_variance            = 0.001;
    remove_low_count_gaussians = true;
    diagonal_loading        = 1.0;
    stigma                  = 1.0;
    variance_lrate          = 1.0;
  }
  void Register(ParseOptions *po) {
    std::string module = "MleDiagGmmOptions: ";
    po->Register("min-gaussian-weight", &min_gaussian_weight,
                 module+"Min Gaussian weight before we remove it.");
    po->Register("min-gaussian-occupancy", &min_gaussian_occupancy,
                 module+"Minimum occupancy to update a Gaussian.");
    po->Register("min-variance", &min_variance,
                 module+"Variance floor (absolute variance).");
    po->Register("remove-low-count-gaussians", &remove_low_count_gaussians,
                 module+"If true, remove Gaussians that fall below the floors.");
    po->Register("diagonal-loading", &diagonal_loading, module+"Ensure the negative definiteness for the Hessian.");
    po->Register("stigma", &stigma, module+"Limit the change of the log variance.");
    po->Register("variance-lrate", &variance_lrate, module+"Learning rate for the variance.");
  }
};

class VtsAccumDiagGmm {
 public:
  VtsAccumDiagGmm()
      : dim_(0),
        num_cepstral_(0),
        num_comp_(0),
        flags_(0) {
  }
  ;

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Allocates memory for accumulators
  void Resize(int32 num_comp, int32 dim, int32 num_cepstral,
              GmmFlagsType flags);
  void Resize(const DiagGmm &gmm, int32 num_cepstral, GmmFlagsType flags);

  /// Returns the number of mixture components
  int32 NumGauss() const {
    return num_comp_;
  }
  /// Returns the number of cepstral components
  int32 NumCepstral() const {
    return num_cepstral_;
  }
  /// Returens the dimenstionality of the feature vectors
  int32 Dim() const {
    return dim_;
  }

  /// Accumulate mean static stats for all component, the posterior is already
  /// in the input stats
  BaseFloat AccumulateFromDiag(const DiagGmm &gmm_clean,
                               const DiagGmm &gmm_noisy,
                               const std::vector<Matrix<double> > &Jx,
                               int32 offset,
                               const VectorBase<BaseFloat> &data,
                               BaseFloat weight);

  void SetZero(GmmFlagsType flags);
  void Scale(BaseFloat f, GmmFlagsType flags);

  /// Increment with stats from this other accumulator (times scale)
  void Add(double scale, const VtsAccumDiagGmm &acc);

  // const accessors
  const GmmFlagsType Flags() const {
    return flags_;
  }
  const Vector<double> &occupancy() const {
    return occupancy_;
  }
  const MatrixBase<double> &mu_vs() const {
    return mu_vs_;
  }
  const MatrixBase<double> &mu_vd() const {
    return mu_vd_;
  }
  const MatrixBase<double> &mu_va() const {
    return mu_va_;
  }
  const MatrixBase<double> &mu_ms() const {
    return mu_ms_;
  }
  const MatrixBase<double> &mu_md() const {
    return mu_md_;
  }
  const MatrixBase<double> &mu_ma() const {
    return mu_ma_;
  }
  const MatrixBase<double> &var_js() const {
    return var_js_;
  }
  const MatrixBase<double> &var_jd() const {
    return var_jd_;
  }
  const MatrixBase<double> &var_ja() const {
    return var_ja_;
  }
  const MatrixBase<double> &var_hs() const {
    return var_hs_;
  }
  const MatrixBase<double> &var_hd() const {
    return var_hd_;
  }
  const MatrixBase<double> &var_ha() const {
    return var_ha_;
  }

 private:
  int32 dim_;
  int32 num_cepstral_;
  int32 num_comp_;
  /// Flags corresponding to the accumulators
  GmmFlagsType flags_;

  /// weight statistics
  Vector<double> occupancy_;
  /// mean statistics for static(s), delta(d) and accelerate(a) features
  Matrix<double> mu_vs_, mu_vd_, mu_va_;  // the vector part
  Matrix<double> mu_ms_, mu_md_, mu_ma_;  // the matrix part
  /// variance statistics for s, d and a
  Matrix<double> var_js_, var_jd_, var_ja_;  // the diagonal Jacobian
  Matrix<double> var_hs_, var_hd_, var_ha_;  // the Hessian matrix

};

/// Returns "augmented" version of flags: e.g. if just updating means, need
/// weights too.
GmmFlagsType AugmentGmmFlags(GmmFlagsType f);

inline void VtsAccumDiagGmm::Resize(const DiagGmm &gmm, int32 num_cepstral,
                                    GmmFlagsType flags) {
  Resize(gmm.NumGauss(), gmm.Dim(), num_cepstral, flags);
}

/// Estimate the parameters of Gaussian mixture model
void VtsDiagGmmUpdate(const VtsDiagGmmOptions &config,
                      const VtsAccumDiagGmm &diaggmm_acc,
                      GmmFlagsType flags,
                      DiagGmm *gmm,
                      BaseFloat *obj_chagne_out,
                      BaseFloat *count_out);

}  // End namespace kaldi

#endif /* VTS_VTS_ACCUM_DIAG_GMM_H_ */
