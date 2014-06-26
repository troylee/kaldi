/*
 * vts/vts-accum-am-diag-gmm.cc
 *
 *  Created on: Nov 23, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "gmm/am-diag-gmm.h"
#include "util/stl-utils.h"
#include "vts/vts-first-order.h"
#include "vts/vts-accum-am-diag-gmm.h"

namespace kaldi {

VtsAccumAmDiagGmm::~VtsAccumAmDiagGmm() {
  DeletePointers(&gmm_accumulators_);
}

void VtsAccumAmDiagGmm::Init(const AmDiagGmm &model, int32 num_cepstral,
                             GmmFlagsType flags) {
  DeletePointers(&gmm_accumulators_);  // in case was non-empty
  gmm_accumulators_.resize(model.NumPdfs(), NULL);
  for (int32 i = 0; i < model.NumPdfs(); ++i) {
    gmm_accumulators_[i] = new VtsAccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i), num_cepstral, flags);
  }
}

void VtsAccumAmDiagGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_accumulators_.size(); i++) {
    gmm_accumulators_[i]->SetZero(flags);
  }
}

void VtsAccumAmDiagGmm::Read(std::istream &in_stream, bool binary, bool add) {
  int32 num_pdfs;
  ExpectToken(in_stream, binary, "<NUMPDFS>");
  ReadBasicType(in_stream, binary, &num_pdfs);
  KALDI_ASSERT(num_pdfs > 0);
  if (!add || (add && gmm_accumulators_.empty())) {
    gmm_accumulators_.resize(num_pdfs, NULL);
    for (std::vector<VtsAccumDiagGmm*>::iterator it = gmm_accumulators_.begin(),
        end = gmm_accumulators_.end(); it != end; ++it) {
      if (*it != NULL)
        delete *it;
      *it = new VtsAccumDiagGmm();
      (*it)->Read(in_stream, binary, add);
    }
  } else {
    if (gmm_accumulators_.size() != static_cast<size_t>(num_pdfs)) {
      KALDI_ERR<< "Adding accumulators but num-pdfs do not match: "
      << (gmm_accumulators_.size()) << " vs. "
      << (num_pdfs);
    }
    for(std::vector<VtsAccumDiagGmm*>::iterator it = gmm_accumulators_.begin(), end=gmm_accumulators_.end(); it!=end; ++it) {
      (*it)->Read(in_stream, binary, add);
    }
  }

  in_stream.peek();
  if (!in_stream.eof()) {
    double like, frames;
    ExpectToken(in_stream, binary, "<total_like>");
    ReadBasicType(in_stream, binary, &like);
    total_log_like_ = (add) ? total_log_like_ + like : like;
    ExpectToken(in_stream, binary, "<total_frames>");
    ReadBasicType(in_stream, binary, &frames);
    total_frames_ = (add) ? total_frames_ + frames : frames;
  }
}

void VtsAccumAmDiagGmm::Write(std::ostream &out_stream, bool binary) const {
  int32 num_pdfs = gmm_accumulators_.size();
  WriteToken(out_stream, binary, "<NUMPDFS>");
  WriteBasicType(out_stream, binary, num_pdfs);
  for (std::vector<VtsAccumDiagGmm*>::const_iterator it = gmm_accumulators_
      .begin(), end = gmm_accumulators_.end(); it != end; ++it) {
    (*it)->Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "<total_like>");
  WriteBasicType(out_stream, binary, total_log_like_);

  WriteToken(out_stream, binary, "<total_frames>");
  WriteBasicType(out_stream, binary, total_frames_);
}


BaseFloat VtsAccumAmDiagGmm::AccumulateForGmm(const AmDiagGmm &model_clean,
                             const AmDiagGmm &model_noisy,
                             const std::vector<Matrix<double> > &Jx,
                             const VectorBase<BaseFloat> &data, int32 gmm_index,
                             BaseFloat weight) {
  KALDI_ASSERT(gmm_index >= 0 && static_cast<size_t>(gmm_index) < gmm_accumulators_.size() )

    int32 offset = GetGaussianOffset(model_clean, gmm_index);
  BaseFloat log_like = gmm_accumulators_[gmm_index]->AccumulateFromDiag(model_clean.GetPdf(gmm_index),
                                                   model_noisy.GetPdf(gmm_index),
                                                   Jx, offset, data, weight);

  total_log_like_ += log_like * weight;
  total_frames_ += weight;

  return log_like;

}

BaseFloat VtsAccumAmDiagGmm::TotStatsCount() const {
  double ans = 0.0;
  for (int32 i = 0; i < NumAccs(); i++) {
    const VtsAccumDiagGmm &acc = GetAcc(i);
    ans += acc.occupancy().Sum();
  }
  return ans;
}

const VtsAccumDiagGmm& VtsAccumAmDiagGmm::GetAcc(int32 index) const {
  KALDI_ASSERT(
      index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

VtsAccumDiagGmm& VtsAccumAmDiagGmm::GetAcc(int32 index) {
  KALDI_ASSERT(
      index >= 0 && index < static_cast<int32>(gmm_accumulators_.size()));
  return *(gmm_accumulators_[index]);
}

void VtsAccumAmDiagGmm::Add(BaseFloat scale, const VtsAccumAmDiagGmm &other) {
  total_frames_ += scale * other.total_frames_;
  total_log_like_ += scale * other.total_log_like_;

  int32 num_accs = NumAccs();
  KALDI_ASSERT(num_accs == other.NumAccs());
  for (int32 i = 0; i < num_accs; ++i) {
    gmm_accumulators_[i]->Add(scale, *(other.gmm_accumulators_[i]));
  }
}

void VtsAccumAmDiagGmm::Scale(BaseFloat scale) {
  for (int32 i = 0; i < NumAccs(); ++i) {
    VtsAccumDiagGmm &acc = GetAcc(i);
    acc.Scale(scale, acc.Flags());
  }
  total_frames_ *= scale;
  total_log_like_ *= scale;
}

void VtsAmDiagGmmUpdate(const VtsDiagGmmOptions &config,
                        const VtsAccumAmDiagGmm &amdiaggmm_acc,
                        GmmFlagsType flags,
                        AmDiagGmm *am_gmm,
                        BaseFloat *obj_change_out,
                        BaseFloat *count_out) {
  if (amdiaggmm_acc.Dim() != am_gmm->Dim()) {
    KALDI_ERR<< "Dimensions of accumulator " << amdiaggmm_acc.Dim()
    << " and gmm " << am_gmm->Dim() << " do not match!";
  }

  KALDI_ASSERT(am_gmm != NULL);
  KALDI_ASSERT(amdiaggmm_acc.NumAccs() == am_gmm->NumPdfs());
  if(obj_change_out != NULL) *obj_change_out = 0.0;
  if(count_out != NULL) *count_out = 0.0;
  BaseFloat tmp_obj_change, tmp_count;
  BaseFloat *p_obj = (obj_change_out !=NULL)? &tmp_obj_change:NULL,
  *p_count = (count_out != NULL)? &tmp_count : NULL;

  for(size_t i =0; i<amdiaggmm_acc.NumAccs(); ++i) {
    VtsDiagGmmUpdate(config, amdiaggmm_acc.GetAcc(i), flags, &(am_gmm->GetPdf(i)), p_obj, p_count);

    if(obj_change_out !=NULL) *obj_change_out += tmp_obj_change;
    if(count_out !=NULL) *count_out += tmp_count;
  }

}

}  // End namespace kaldi

