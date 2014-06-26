/*
 * nnet/nnet-robm.cc
 *
 *  Created on: Apr 30, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-math.h"
#include "nnet/nnet-rorbm.h"

namespace kaldi {

typedef kaldi::int32 int32;

void RoRbm::InferBatchsizeChange(int32 batchsize) {
  if (batchsize < 0) {
    KALDI_ERR<< "Batch size much be greater than 0! [batchsize=" << batchsize << "].";
  }

  mu_.Resize(batchsize, vis_dim_);
  mu_hat_.Resize(batchsize, vis_dim_);

  s_.Resize(batchsize, vis_dim_);
  phi_s_.Resize(batchsize, vis_dim_);

  log_sprob_0_.Resize(batchsize, vis_dim_);
  log_sprob_1_.Resize(batchsize, vis_dim_);

  mat_tmp_.Resize(batchsize, vis_dim_);

  v_sample_.Resize(batchsize, vis_dim_);
  v_condmean_.Resize(batchsize, vis_dim_);
  v_condstd_.Resize(batchsize, vis_dim_);

  ha_.Resize(batchsize, clean_hid_dim_);
  haprob_.Resize(batchsize, clean_hid_dim_);

  hs_.Resize(batchsize, noise_hid_dim_);
  hsprob_.Resize(batchsize, noise_hid_dim_);

  z_.Resize(batchsize, vis_dim_);

  batch_size_ = batchsize;

}

void RoRbm::TrainBatchsizeChange(int32 batchsize) {
  if (batchsize < 0) {
    KALDI_ERR<< "Batch size much be greater than 0! [batchsize=" << batchsize << "].";
  }

  fp_v_.Resize(batchsize, vis_dim_);
  fp_vt_.Resize(batchsize, vis_dim_);

  mu_t_hat_.Resize(batchsize, vis_dim_);

  fp_s_.Resize(batchsize, vis_dim_);

  fp_vt_condstd_.Resize(batchsize, vis_dim_);
  fp_vt_condmean_.Resize(batchsize, vis_dim_);

  fp_ha_.Resize(batchsize, clean_hid_dim_);

  fp_hs_.Resize(batchsize, noise_hid_dim_);

  batch_size_=batchsize;

}

void RoRbm::InitInference(int32 batchsize) {

  std_hat_.Resize(vis_dim_);
  inv_gamma2_tmp_.Resize(vis_dim_);

  vec_tmp_.Resize(vis_dim_);

  InferBatchsizeChange(batchsize);
  batch_size_ = batchsize;

}

void RoRbm::InitTraining(int32 batchsize) {

  InitInference(batchsize);

  U_corr_.Resize(noise_hid_dim_, vis_dim_);
  d_corr_.Resize(vis_dim_);
  e_corr_.Resize(noise_hid_dim_);
  bt_corr_.Resize(vis_dim_);
  lamt2_corr_.Resize(vis_dim_);
  gamma2_corr_.Resize(vis_dim_);

  U_pos_.Resize(noise_hid_dim_, vis_dim_);
  U_neg_.Resize(noise_hid_dim_, vis_dim_);
  d_pos_.Resize(vis_dim_);
  d_neg_.Resize(vis_dim_);
  e_pos_.Resize(noise_hid_dim_);
  e_neg_.Resize(noise_hid_dim_);
  bt_pos_.Resize(vis_dim_);
  bt_neg_.Resize(vis_dim_);
  lamt2_pos_.Resize(vis_dim_);
  lamt2_neg_.Resize(vis_dim_);
  gamma2_pos_.Resize(vis_dim_);
  gamma2_neg_.Resize(vis_dim_);

  U_tmp_.Resize(noise_hid_dim_, vis_dim_);

  e_.Resize(noise_hid_dim_);

  vec_tmp2_.Resize(vis_dim_);
  lamt2_hat_.Resize(vis_dim_);

  U_corr_.SetZero();
  d_corr_.SetZero();
  e_corr_.SetZero();
  bt_corr_.SetZero();
  lamt2_corr_.SetZero();
  gamma2_corr_.SetZero();

  TrainBatchsizeChange(batchsize);
  batch_size_ = batchsize;

}

void RoRbm::ReadData(std::istream &is, bool binary) {

  /* Read in the layer types */
  std::string vis_node_type, clean_hid_node_type, noise_hid_node_type;
  ReadToken(is, binary, &vis_node_type);
  ReadToken(is, binary, &clean_hid_node_type);
  ReadToken(is, binary, &noise_hid_node_type);

  KALDI_ASSERT(vis_node_type == "gauss");
  KALDI_ASSERT(clean_hid_node_type == "bern");
  KALDI_ASSERT(noise_hid_node_type == "bern");

  /* Read in the hidden dim for noise RBM */
  ReadBasicType(is, binary, &noise_hid_dim_);
  KALDI_ASSERT(noise_hid_dim_ > 0);

  /* Read clean RBM */
  clean_vis_hid_.Read(is, binary);
  clean_vis_bias_.Read(is, binary);
  clean_hid_bias_.Read(is, binary);
  clean_vis_std_.Read(is, binary);

  KALDI_ASSERT(
      clean_vis_hid_.NumRows() == clean_hid_dim_ && clean_vis_hid_.NumCols() == vis_dim_);
  KALDI_ASSERT(clean_vis_bias_.Dim() == vis_dim_);
  KALDI_ASSERT(clean_hid_bias_.Dim() == clean_hid_dim_);
  KALDI_ASSERT(clean_vis_std_.Dim() == vis_dim_);

  clean_vis_var_.CopyFromVec(clean_vis_std_);
  clean_vis_var_.Power(2.0);

  /* Read Noise RBM */
  U_.Read(is, binary);  // weight matrix
  d_.Read(is, binary);  // visible bias
  e_.Read(is, binary);  // hidden bias

  KALDI_ASSERT(U_.NumRows() == noise_hid_dim_ && U_.NumCols() == vis_dim_);
  KALDI_ASSERT(d_.Dim() == vis_dim_);
  KALDI_ASSERT(e_.Dim() == noise_hid_dim_);

  /* Parameters for noisy inputs */
  bt_.Read(is, binary);
  lamt2_.Read(is, binary);
  gamma2_.Read(is, binary);

  KALDI_ASSERT(bt_.Dim() == vis_dim_);
  KALDI_ASSERT(lamt2_.Dim() == vis_dim_);
  KALDI_ASSERT(gamma2_.Dim() == vis_dim_);
}

void RoRbm::WriteData(std::ostream &os, bool binary) const {

  /* Write layer types */
  // vis type
  WriteToken(os, binary, "gauss");

  // clean hidden type
  WriteToken(os, binary, "bern");

  // noise hidden type
  WriteToken(os, binary, "bern");

  /* Write the hidden dim for noise RBM */
  WriteBasicType(os, binary, noise_hid_dim_);

  /* Write clean RBM */
  clean_vis_hid_.Write(os, binary);
  clean_vis_bias_.Write(os, binary);
  clean_hid_bias_.Write(os, binary);
  clean_vis_std_.Write(os, binary);

  /* Write noise RBM */
  U_.Write(os, binary);
  d_.Write(os, binary);
  e_.Write(os, binary);

  /* Write noisy input parameters */
  bt_.Write(os, binary);
  lamt2_.Write(os, binary);
  gamma2_.Write(os, binary);
}

void RoRbm::Inference(const CuMatrix<BaseFloat> &vt_cn) {

  int32 n = vt_cn.NumRows();
  if (n != batch_size_) {
    /* Resize the necessary variables */
    InferBatchsizeChange(n);
  }

  /* initialize the clean RBM hidden states */
  haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
  haprob_.AddMatMat(1.0, vt_cn, kNoTrans, clean_vis_hid_, kTrans, 1.0);
  cu::Sigmoid(haprob_, &haprob_);
  cu_rand_clean_hid_.BinarizeProbs(haprob_, &ha_);

  /* initialize the noise RBM hidden states */
  cu_rand_noise_hid_.RandUniform(&hs_);

  /* do inference */
  z_.SetZero();

  /* run multiple iterations to denoise */
  for (int32 k = 0; k < num_infer_iters_; ++k) {
    // downsample - from hidden to visible
    /* needed for sprob_0, clean GRBM */
    mu_.AddMatMat(1.0, ha_, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // ha * W
    mu_.MulColsVec(clean_vis_var_);  // var * (ha * W)
    mu_.AddVecToRows(1.0, clean_vis_bias_, 1.0);  // b + var * (ha * W)
    /* needed for sprob_1, noise RBM */
    phi_s_.AddVecToRows(1.0, d_, 0.0);  // d
    phi_s_.AddMatMat(1.0, hs_, kNoTrans, U_, kNoTrans, 1.0);  // d + hs * U

    /* needed for sprob_1, noisy input */
    mu_hat_.CopyFromMat(vt_cn);
    mu_hat_.MulColsVec(gamma2_);  // gamma2 .* vt_cn
    mu_hat_.AddMat(1.0, mu_, 1.0);  // mu + gamma2 .* vt_cn
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.Add(1.0);  // gamma2 + 1
    mu_hat_.DivColsVec(vec_tmp_);  // (mu + gamma2 .* vt_cn) ./ (gamma2 + 1)

    /* needed for sprob_1 */
    vec_tmp_.Power(0.5);  // sqrt(gamma2 + 1)
    std_hat_.CopyFromVec(clean_vis_std_);  // std_vec
    std_hat_.DivElements(vec_tmp_);  // std_vec ./ sqrt(gamma2 + 1)

    /* compute log_sprob_1 */
    log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

    mat_tmp_.CopyFromMat(vt_cn);  // vt_cn
    mat_tmp_.Power(2.0);  // vt_cn.^2
    vec_tmp_.CopyFromVec(gamma2_);  // gamma2
    vec_tmp_.DivElements(clean_vis_var_);  // gamma2 ./ var_vec
    mat_tmp_.MulColsVec(vec_tmp_);  // vt_cn.^2 .* gamma2 ./ var_vec
    log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec

    mat_tmp_.CopyFromMat(mu_hat_);  // mu_hat
    mat_tmp_.DivColsVec(std_hat_);  // mu_hat ./ std_hat
    mat_tmp_.Power(2.0);  // mu_hat.^2 ./ std_hat.^2
    log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

    vec_tmp_.CopyFromVec(std_hat_);  // std_hat
    vec_tmp_.ApplyLog();  // log(std_hat)
    log_sprob_1_.AddVecToRows(1.0, vec_tmp_, 1.0);

    /* compute log_sprob_0 */
    mat_tmp_.CopyFromMat(mu_);  // mu
    mat_tmp_.Power(2.0);  // mu.^2
    mat_tmp_.DivColsVec(clean_vis_var_);  // mu.^2 ./ var_vec
    log_sprob_0_.AddMat(0.5, mat_tmp_, 0.0);  // mu.^2 ./ var_vec

    vec_tmp_.CopyFromVec(clean_vis_std_);  // std_vec
    vec_tmp_.ApplyLog();  // log(std_vec)
    log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 1.0);  // mu.^2 ./ var_vec + log(std_vec)

    /* log(exp(log_sprob_0) + exp(log_sprob_1)) */
    log_sprob_0_.LogAddExpMat(log_sprob_1_);

    /* compute sprob (saved in log_sprob_1) */
    log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1_.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    /* compute s */
    cu_rand_vis_.BinarizeProbs(log_sprob_1_, &s_);

    /* compute v_condmean */
    v_condmean_.CopyFromMat(mu_);  // mu

    mat_tmp_.CopyFromMat(vt_cn);  // vt_cn
    mat_tmp_.MulElements(s_);  // s .* vt_cn
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* s.* vt_cn
    v_condmean_.AddMat(1.0, mat_tmp_, 1.0);  // gamma2 .* s.* vt_cn + mu

    mat_tmp_.CopyFromMat(s_);  // s
    mat_tmp_.MulColsVec(gamma2_);  // gamma2 .* s
    mat_tmp_.Add(1.0);  // gamma2 .* s + 1
    v_condmean_.DivElements(mat_tmp_);  // (gamma2 .* s.* vt_cn + mu) ./ (gamma2 .* s + 1)

    /* compute v_condstd */
    v_condstd_.AddVecToRows(1.0, clean_vis_std_, 0.0);  // std_vec
    mat_tmp_.Power(0.5);  // sqrt(gamma2 .* s + 1)
    v_condstd_.DivElements(mat_tmp_);  // std_vec ./ sqrt(gamma2 .* s + 1)

    /* sample vt_cn */
    cu_rand_vis_.RandGaussian(&v_sample_);
    v_sample_.MulElements(v_condstd_);
    v_sample_.AddMat(1.0, v_condmean_, 1.0);

    /* sample the hidden variables */
    haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
    haprob_.AddMatMat(1.0, v_sample_, kNoTrans, clean_vis_hid_, kTrans, 1.0);  // v*W + c
    cu::Sigmoid(haprob_, &haprob_);  // 1.0 ./ (1.0 + exp(v*W + c))
    cu_rand_clean_hid_.BinarizeProbs(haprob_, &ha_);  // binarize

    hsprob_.AddVecToRows(1.0, e_, 0.0);  // e
    hsprob_.AddMatMat(1.0, s_, kNoTrans, U_, kTrans, 1.0);  // s*U + e
    cu::Sigmoid(hsprob_, &hsprob_);  // 1.0 ./ (1.0 + exp(s*U + e))
    cu_rand_noise_hid_.BinarizeProbs(hsprob_, &hs_);  // binarize

    /* collect smooth estimates */
    if (z_start_iter_ >= 0) {  // negative z indicates no collection
      if (k == z_start_iter_) {
        z_.CopyFromMat(v_condmean_);
      } else if (k > z_start_iter_) {
        z_.AddMat(1 - z_momentum_, v_condmean_, z_momentum_);
      }
    }

  }  // end iteration k

}

void RoRbm::GetReconstruction(CuMatrix<BaseFloat> *v) {
  /* using smoother version rather than the v_sample_ */
  v->CopyFromMat(v_condmean_);
}

void RoRbm::CollectPositiveStats(const CuMatrix<BaseFloat> &vt_cn,
                                 CuVector<BaseFloat> *s_mu) {
  /* positive phase gradient */
  mat_tmp_.CopyFromMat(vt_cn);  // vt_cn
  mat_tmp_.MulColsVec(lamt2_);  // vt_cn .* lamt2
  bt_pos_.AddRowSumMat(1.0, mat_tmp_, 0.0);  // sum(vt_cn .* lamt2)

  mat_tmp_.AddVecToRows(1.0, bt_, 0.0);  // bt
  mat_tmp_.AddMat(-0.5, vt_cn, 1.0);  // -0.5 * vt_cn + bt
  mat_tmp_.MulElements(vt_cn);  // -0.5 * vt_cn.*vt_cn + vt_cn .* bt
  lamt2_pos_.AddRowSumMat(1.0, mat_tmp_, 0.0);  // sum(-0.5 * vt_cn.^2 + vt_cn .* bt)

  mat_tmp_.CopyFromMat(vt_cn);  // vt_cn
  /* use v_condmean_ instead of v_sample_, i.e. no randomness */
  mat_tmp_.AddMat(1.0, v_condmean_, -1.0);  // v - vt_cn
  mat_tmp_.Power(2.0);  // (v - vt_cn).^2
  mat_tmp_.MulElements(s_);  // s .* (v - vt_cn).^2
  mat_tmp_.Scale(-0.5);  // -0.5 * s.* (v - vt_cn).^2
  gamma2_pos_.AddRowSumMat(1.0, mat_tmp_, 0.0);  // sum(-0.5 * s.* (v - vt_cn).^2)
  gamma2_pos_.DivElements(clean_vis_var_);  // sum(-0.5 * s.* (v - vt_cn).^2) ./ var_vec

  /* accumulate average statistics for s_ */
  s_mu->AddRowSumMat(0.05 / s_.NumRows(), s_, 0.95);

  mat_tmp_.CopyFromMat(s_);  // s
  mat_tmp_.AddVecToRows(-1.0, *s_mu, 1.0);  // s - s_mu
  U_pos_.AddMatMat(1.0, hs_, kTrans, mat_tmp_, kNoTrans, 0.0);
  d_pos_.AddRowSumMat(1.0, mat_tmp_, 0.0);
  e_pos_.AddRowSumMat(1.0, hs_, 0.0);

}

/*
 * One iteration of stochastic approximation procedure
 */
void RoRbm::SAPIteration() {

  /* #1. p(s|hs, ha, vt) */
  mu_.AddMatMat(1.0, fp_ha_, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // fp_ha * W
  mu_.MulColsVec(clean_vis_var_);// (fp_ha * W) .* var_vec
  mu_.AddVecToRows(1.0, clean_vis_bias_, 1.0);// (fp_ha * W) .* var_vec + b

  phi_s_.AddVecToRows(1.0, d_, 0.0);// d
  phi_s_.AddMatMat(1.0, fp_hs_, kNoTrans, U_, kNoTrans, 1.0);// fp_hs * U + d

  mu_hat_.CopyFromMat(fp_vt_);// fp_vt
  mu_hat_.MulColsVec(gamma2_);// fp_vt .* gamma2
  mu_hat_.AddMat(1.0, mu_, 1.0);// mu + fp_vt .* gamma2
  vec_tmp_.CopyFromVec(gamma2_);// gamma2
  vec_tmp_.Add(1.0);// gamma2 + 1
  mu_hat_.DivColsVec(vec_tmp_);// (mu + fp_vt .* gamma2) ./ (gamma2 + 1)

  std_hat_.CopyFromVec(clean_vis_std_);// std_vec
  vec_tmp_.Power(0.5);// 1.0 / sqrt(gamma2 + 1)
  std_hat_.DivElements(vec_tmp_);// std_vec ./ sqrt(gamma2 + 1)

  /** compute log_sprob_1 **/
  log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

  mat_tmp_.CopyFromMat(fp_vt_);// fp_vt
  mat_tmp_.Power(2);// fp_vt.^2
  mat_tmp_.MulColsVec(gamma2_);// gamma2 .* (fp_vt.^2)
  mat_tmp_.DivColsVec(clean_vis_var_);// gamma2 .* (fp_vt.^2) ./ var_vec
  log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);// phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec

  mat_tmp_.CopyFromMat(mu_hat_);// mu_hat
  mat_tmp_.DivColsVec(std_hat_);// mu_hat ./ std_hat
  mat_tmp_.Power(2.0);// mu_hat.^2 ./ std_hat.^2
  log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);// phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

  vec_tmp_.CopyFromVec(std_hat_);// std_hat
  vec_tmp_.ApplyLog();// log(std_hat)
  log_sprob_1_.AddVecToRows(1.0, vec_tmp_, 1.0);//  phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2 + log(std_hat)

  /** compute log_sprob_0 **/
  log_sprob_0_.CopyFromMat(mu_);  // mu
  log_sprob_0_.Power(2.0);// mu.^2
  log_sprob_0_.DivColsVec(clean_vis_var_);// mu.^2 ./ var_vec
  log_sprob_0_.Scale(0.5);// 0.5 * mu.^2 ./ var_vec
  vec_tmp_.CopyFromVec(clean_vis_std_);// std_vec
  vec_tmp_.ApplyLog();// log(std_vec)
  log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 1.0);

  /** log(exp(log_sprob_0) + exp(log_sprob_1)) **/
  log_sprob_0_.LogAddExpMat(log_sprob_1_);

  /** compute sprob (saved in log_sprob_1) **/
  log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
  log_sprob_1_.ApplyExp();// exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

  /** compute s **/
  cu_rand_vis_.BinarizeProbs(log_sprob_1_, &fp_s_);

  /* #2. p(v|s, ha, vt) */
  mat_tmp_.CopyFromMat(fp_s_);  // fp_s
  mat_tmp_.MulColsVec(gamma2_);// gamma2 .* fp_s
  v_condmean_.CopyFromMat(mat_tmp_);// gamma2 .* fp_s
  v_condmean_.MulElements(fp_vt_);// gamma2 .* fp_s .* fp_vt
  v_condmean_.AddMat(1.0, mu_, 1.0);// gamma2 .* fp_s .* fp_vt + mu
  mat_tmp_.Add(1.0);// gamma2 .* fp_s + 1.0
  v_condmean_.DivElements(mat_tmp_);// (gamma2 .* fp_s .* fp_vt + mu) ./ (gamma2 .* fp_s + 1.0)

  v_condstd_.CopyFromMat(mat_tmp_);// gamma2 .* fp_s + 1.0
  v_condstd_.Power(0.5);// sqrt(gamma2 .* fp_s + 1.0)
  v_condstd_.InvertElements();// 1.0 ./ sqrt(gamma2 .* fp_s + 1.0)
  v_condstd_.MulColsVec(clean_vis_std_);// std_vec ./ sqrt(gamma2 .* fp_s + 1.0)

  /** sample from v **/
  cu_rand_vis_.RandGaussian(&fp_v_);  // random
  fp_v_.MulElements(v_condstd_);// fp_v .* v_condstd
  fp_v_.AddMat(1.0, v_condmean_, 1.0);// fp_v .* v_condstd + v_condmean

  /* #3. p(s|v, hs) */
  vec_tmp_.CopyFromVec(clean_vis_var_);  // var_vec
  vec_tmp_.MulElements(bt_);// var_vec .* bt
  vec_tmp2_.CopyFromVec(gamma2_);// gamma2
  vec_tmp2_.DivElements(lamt2_);// gamma2 ./ lamt2
  mu_t_hat_.CopyFromMat(fp_v_);// fp_v
  mu_t_hat_.MulColsVec(vec_tmp2_);// (gamma2 ./ lamt2) .* fp_v
  mu_t_hat_.AddVecToRows(1.0, vec_tmp_, 1.0);// var_vec .* bt + (gamma2 ./ lamt2) .* fp_v
  vec_tmp2_.AddVec(1.0, clean_vis_var_, 1.0);// var_vec + gamma2 ./ lamt2
  mu_t_hat_.DivColsVec(vec_tmp2_);// (var_vec .* bt + (gamma2 ./ lamt2) .* fp_v) ./ (var_vec + gamma2 ./ lamt2)

  lamt2_hat_.CopyFromVec(vec_tmp2_);// var_vec + gamma2 ./ lamt2
  lamt2_hat_.DivElements(clean_vis_var_);// (var_vec + gamma2 ./ lamt2) ./ var_vec
  lamt2_hat_.MulElements(lamt2_);// (var_vec + gamma2 ./ lamt2) ./ (var_vec ./ lamt2)

  /** compute log_sprob_1 **/
  log_sprob_1_.CopyFromMat(phi_s_);  // phi_s

  mat_tmp_.CopyFromMat(fp_v_);// fp_v
  mat_tmp_.Power(2);// fp_v.^2
  mat_tmp_.MulColsVec(gamma2_);// gamma2 .* (fp_v.^2)
  mat_tmp_.DivColsVec(clean_vis_var_);// gamma2 .* (fp_v.^2) ./ var_vec
  log_sprob_1_.AddMat(-0.5, mat_tmp_, 1.0);// phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec

  mat_tmp_.CopyFromMat(mu_t_hat_);// mu_t_hat
  mat_tmp_.Power(2.0);// mu_t_hat.^2
  mat_tmp_.MulColsVec(lamt2_hat_);// (mu_t_hat.^2) .* lamt2_hat
  log_sprob_1_.AddMat(0.5, mat_tmp_, 1.0);// phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec + 0.5 * mu_t_hat.^2 .* lamt2_hat

  vec_tmp_.CopyFromVec(lamt2_hat_);// lamt2_hat
  vec_tmp_.ApplyLog();// log(lamt2_hat)
  log_sprob_1_.AddVecToRows(-0.5, vec_tmp_, 1.0);//  phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec + 0.5 * mu_t_hat.^2 .* lamt2_hat - log(sqrt(lamt2_hat))

  /** compute log_sprob_0 **/
  vec_tmp_.CopyFromVec(bt_);  // bt
  vec_tmp_.Power(2.0);// bt.^2
  vec_tmp_.MulElements(lamt2_);// bt.^2 .* lamt2
  vec_tmp2_.CopyFromVec(lamt2_);// lamt2
  vec_tmp2_.ApplyLog();// log(lamt2)
  vec_tmp_.AddVec(-0.5, vec_tmp2_, 0.5);// 0.5 * bt.^2 .* lamt2 - log(sqrt(lmat2))
  log_sprob_0_.AddVecToRows(1.0, vec_tmp_, 0.0);

  /** log(exp(log_sprob_0) + exp(log_sprob_1)) **/
  log_sprob_0_.LogAddExpMat(log_sprob_1_);

  /** compute sprob (saved in log_sprob_1) **/
  log_sprob_1_.AddMat(-1.0, log_sprob_0_);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
  log_sprob_1_.ApplyExp();// exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

  cu_rand_vis_.BinarizeProbs(log_sprob_1_, &fp_s_);

  /* #4. p(vt | s, v) */
  vec_tmp_.CopyFromVec(gamma2_);  // gamma2
  vec_tmp_.DivElements(lamt2_);// gamma2 ./ lamt2
  mat_tmp_.CopyFromMat(fp_s_);// fp_s
  mat_tmp_.MulColsVec(vec_tmp_);// fp_s .* (gamma2 ./ lamt2)
  vec_tmp_.CopyFromVec(clean_vis_var_);// var_vec
  vec_tmp_.MulElements(bt_);// var_vec .* bt
  fp_vt_condmean_.CopyFromMat(mat_tmp_);// fp_s .* (gamma2 ./ lamt2)
  fp_vt_condmean_.MulElements(fp_v_);// fp_s .* (gamma2 ./ lamt2) .* fp_v
  fp_vt_condmean_.AddVecToRows(1.0, vec_tmp_, 1.0);// var_vec .* bt + fp_s .* (gamma2 ./ lamt2) .* fp_v
  mat_tmp_.AddVecToRows(1.0, clean_vis_var_, 1.0);// var_vec + fp_s .* (gamma2 ./ lamt2)
  fp_vt_condmean_.DivElements(mat_tmp_);// (var_vec .* bt + fp_s .* (gamma2 ./ lamt2) .* fp_v) ./ (var_vec + fp_s .* (gamma2 ./ lamt2))

  vec_tmp_.CopyFromVec(clean_vis_var_);// var_vec
  vec_tmp_.DivElements(lamt2_);// var_vec ./ lamt2
  fp_vt_condstd_.AddVecToRows(1.0, vec_tmp_, 0.0);// var_vec ./ lamt2
  fp_vt_condstd_.DivElements(mat_tmp_);// (var_vec ./ lamt2) ./ (var_vec + fp_s .* (gamma2 ./ lamt2))
  fp_vt_condstd_.Power(0.5);// sqrt((var_vec ./ lamt2) ./ (var_vec + fp_s .* (gamma2 ./ lamt2)))

  /** sample from fp_vt_ **/
  cu_rand_vis_.RandGaussian(&fp_vt_);
  fp_vt_.MulElements(fp_vt_condstd_);  // fp_vt .* fp_vt_condstd
  fp_vt_.AddMat(1.0, fp_vt_condmean_, 1.0);// fp_vt .* fp_vt_condstd + fp_vt_condmean

  /* #5. p(hs|s); p(ha|v) */
  haprob_.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
  haprob_.AddMatMat(1.0, fp_v_, kNoTrans, clean_vis_hid_, kTrans, 1.0);// fp_v * W' + c
  cu::Sigmoid(haprob_, &haprob_);
  cu_rand_clean_hid_.BinarizeProbs(haprob_, &fp_ha_);

  hsprob_.AddVecToRows(1.0, e_, 0.0);// e
  hsprob_.AddMatMat(1.0, fp_s_, kNoTrans, U_, kTrans, 1.0);// fp_s * U' + e
  cu::Sigmoid(hsprob_, &hsprob_);
  cu_rand_noise_hid_.BinarizeProbs(hsprob_, &fp_hs_);
}

void RoRbm::CollectNegativeStats(const CuVector<BaseFloat> &s_mu) {
  /* negative phase gradients */
  mat_tmp_.CopyFromMat(fp_vt_);  // fp_vt
  mat_tmp_.MulColsVec(lamt2_);  // fp_vt .* lamt2
  bt_neg_.AddRowSumMat(1.0, mat_tmp_, 0.0);

  mat_tmp_.AddVecToRows(1.0, bt_, 0.0);  // bt
  mat_tmp_.AddMat(-0.5, fp_vt_, 1.0);  // -0.5 * fp_vt + bt
  mat_tmp_.MulElements(fp_vt_);  // -0.5 * fp_vt.^2 + fp_vt .* bt
  lamt2_neg_.AddRowSumMat(1.0, mat_tmp_, 0.0);

  mat_tmp_.CopyFromMat(fp_v_);  // fp_v
  mat_tmp_.AddMat(-1.0, fp_vt_, 1.0);  // fp_v - fp_vt
  mat_tmp_.Power(2.0);  // (fp_v - fp_vt).^2
  mat_tmp_.MulElements(fp_s_);  // fp_s .* (fp_v - fp_vt).^2
  mat_tmp_.DivColsVec(clean_vis_var_);  // fp_s .* (fp_v - fp_vt).^2 ./ var_vec
  gamma2_neg_.AddRowSumMat(-0.5, mat_tmp_, 0.0);  // -0.5 * fp_s .* (fp_v - fp_vt).^2 ./ var_vec

  mat_tmp_.CopyFromMat(fp_s_);  // fp_s
  mat_tmp_.AddVecToRows(-1.0, s_mu, 1.0);  // fp_s - s_mu
  U_neg_.AddMatMat(1.0, fp_hs_, kTrans, mat_tmp_, kNoTrans, 0.0);  // (fp_s - s_mu)' * fp_hs
  d_neg_.AddRowSumMat(1.0, mat_tmp_, 0.0);
  e_neg_.AddRowSumMat(1.0, fp_hs_, 0.0);

}

void RoRbm::RoRbmUpdate() {
  BaseFloat lr = learn_rate_ / batch_size_;
  BaseFloat wc = -learn_rate_ * l2_penalty_;

  bt_pos_.AddVec(-1.0, bt_neg_, 1.0);  // bt_pos - bt_neg
  bt_corr_.AddVec(lr, bt_pos_, momentum_);  // momentum * bt_inc + epsilon/n * (bt_pos - bt_neg)
  bt_corr_.AddVec(wc, bt_, 1.0);  // momentum * bt_inc + epsilon/n * (bt_pos - bt_neg) - epsilon * wtcost * bt

  lamt2_pos_.AddVec(-1.0, lamt2_neg_, 1.0);  // lamt2_pos - lamt2_neg
  lamt2_corr_.AddVec(lr, lamt2_pos_, momentum_);
  lamt2_corr_.AddVec(wc, lamt2_, 1.0);

  gamma2_pos_.AddVec(-1.0, gamma2_neg_, 1.0);
  gamma2_corr_.AddVec(0.1 * lr, gamma2_pos_, momentum_);  // gamma2 has relative small learn rate
  gamma2_corr_.AddVec(0.1 * wc, gamma2_, 1.0);  // gamma2 has relative small learn rate

  d_pos_.AddVec(-1.0, d_neg_, 1.0);
  d_corr_.AddVec(lr, d_pos_, momentum_);

  e_pos_.AddVec(-1.0, e_neg_, 1.0);
  e_corr_.AddVec(lr, e_pos_, momentum_);

  U_pos_.AddMat(-1.0, U_neg_, 1.0);
  U_corr_.AddMat(lr, U_pos_, momentum_);
  U_corr_.AddMat(wc, U_, 1.0);

  bt_.AddVec(1.0, bt_corr_, 1.0);
  lamt2_.AddVec(1.0, lamt2_corr_, 1.0);
  gamma2_.AddVec(1.0, gamma2_corr_, 1.0);
  d_.AddVec(1.0, d_corr_, 1.0);
  e_.AddVec(1.0, e_corr_, 1.0);
  U_.AddMat(1.0, U_corr_, 1.0);

  gamma2_.ApplyFloor(0.0);
  lamt2_.ApplyFloor(0.0);
}

}  // end namespace

