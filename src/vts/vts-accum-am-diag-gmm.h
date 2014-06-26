/*
 * vts/vts-accum-am-diag-gmm.h
 *
 *  Created on: Nov 23, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Class for VTS Adaptative training to accumulate sufficient
 *  statistic for updating the GMM models.
 *
 *  Currently only supports MFCC_0_D_A features.
 *
 */

#ifndef VTS_VTS_ACCUM_AM_DIAG_GMM_H_
#define VTS_VTS_ACCUM_AM_DIAG_GMM_H_

#include <vector>

#include "gmm/am-diag-gmm.h"
#include "vts/vts-accum-diag-gmm.h"

namespace kaldi {

class VtsAccumAmDiagGmm {
 public:
  VtsAccumAmDiagGmm()
      : total_frames_(0.0),
        total_log_like_(0.0) {
  }
  ;
  ~VtsAccumAmDiagGmm();

  void Read(std::istream &in_stream, bool binary, bool add = false);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Initializes accumulators for each GMM based on the number of components
  void Init(const AmDiagGmm &model, int32 num_cepstral, GmmFlagsType flags);
  void SetZero(GmmFlagsType flags);

  int32 Dim() const {
    return (
        gmm_accumulators_.empty() || !gmm_accumulators_[0] ?
            0 : gmm_accumulators_[0]->Dim());
  }

  int32 NumAccs() {
    return gmm_accumulators_.size();
  }
  int32 NumAccs() const {
    return gmm_accumulators_.size();
  }

  const VtsAccumDiagGmm& GetAcc(int32 index) const;
  VtsAccumDiagGmm& GetAcc(int32 index);

  void Add(BaseFloat scale, const VtsAccumAmDiagGmm &other);
  void Scale(BaseFloat scale);

  /// Accumulate stats for a single GMM in the model; returns log likelihood
  BaseFloat AccumulateForGmm(const AmDiagGmm &model_clean,
                             const AmDiagGmm &model_noisy,
                             const std::vector<Matrix<double> > &Jx,
                             const VectorBase<BaseFloat> &data, int32 gmm_index,
                             BaseFloat weight);

  BaseFloat TotStatsCount() const; // returns the total count got by summing the count
    // of the actual stats, may differ from TotCount() if e.g. you did I-smoothing.

    // Be careful since total_frames_ is not updated in AccumulateForGaussian
    BaseFloat TotCount() const { return total_frames_; }
    BaseFloat TotLogLike() const { return total_log_like_; }

 private:
  /// accumulators and update methods for the GMMs
  std::vector<VtsAccumDiagGmm*> gmm_accumulators_;

  /// Totoal counts & likelihood (for diagnostics)
  double total_frames_, total_log_like_;

  // Cannot have copy constructor and assignment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(VtsAccumAmDiagGmm);
};

void VtsAmDiagGmmUpdate(const VtsDiagGmmOptions &config,
                        const VtsAccumAmDiagGmm &amdiaggmm_acc,
                        GmmFlagsType flags,
                        AmDiagGmm *am_gmm,
                        BaseFloat *obj_change_out,
                        BaseFloat *count_out);

}  // End namespace kaldi

#endif /* VTS_VTS_ACCUM_AM_DIAG_GMM_H_ */
