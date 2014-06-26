/*
 * nnet/nnet-rorbm.h
 *
 *  Created on: Apr 30, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef KALDI_NNET_RORBM_H_
#define KALDI_NNET_RORBM_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-rbm.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {
/*
 * Robust RBM
 *
 * The clean GRBM model parameters are pre-trained separately as a conventional
 * GRBM and simply copied here.
 *
 */
class RoRbm : public RbmBase {
 public:
  /*
   * dim_in: the input dimension of the noisy signal;
   * dim_out: the hidden dimension of the clean speech RBM;
   */
  RoRbm(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : RbmBase(dim_in, dim_out, nnet),
        vis_dim_(dim_in),
        clean_hid_dim_(dim_out),
        noise_hid_dim_(0),
        z_momentum_(0.0),
        batch_size_(0),
        first_data_bunch_(true)
  {
    z_start_iter_ = -1;

    vis_type_ = RbmBase::GAUSSIAN;
    clean_hid_type_ = RbmBase::BERNOULLI;
    noise_hid_type_ = RbmBase::BERNOULLI;
  }

  ~RoRbm() {
  }

  ComponentType GetType() const {
    return kRoRbm;
  }

  /* Read model from file */
  void ReadData(std::istream &is, bool binary);

  /* Write model to file */
  void WriteData(std::ostream &os, bool binary) const;

  /*
   * Visible unit type for the clean speech RBM.
   */
  RbmNodeType VisType() const {
    return vis_type_;
  }

  /*
   * Hidden unit type for the clean speech RBM.
   */
  RbmNodeType HidType() const {
    return clean_hid_type_;
  }

  RbmNodeType CleanHidType() const {
    return clean_hid_type_;
  }

  RbmNodeType NoiseHidType() const {
    return noise_hid_type_;
  }

  int32 VisDim() const {
    return vis_dim_;
  }

  int32 CleanHidDim() const {
    return clean_hid_dim_;
  }

  int32 NoiseHidDim() const {
    return noise_hid_dim_;
  }

  void ConvertNoiseHidBias(const CuVector<BaseFloat> &s_mu);

  /*
   * To do posterior inference in a single object RoRbm conditioned on a vt_cn image
   */
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    Inference(in);
    out->CopyFromMat(ha_);
  }

  void Inference(const CuMatrix<BaseFloat> &vt_cn);

  void GetReconstruction(CuMatrix<BaseFloat> *v);

  void CollectPositiveStats(const CuMatrix<BaseFloat> &vt_cn,
                            CuVector<BaseFloat> *s_mu);

  void SAPIteration();

  void CollectNegativeStats(const CuVector<BaseFloat> &s_mu);

  void RoRbmUpdate();

  void InitInference(int32 batchsize);

  void InitTraining(int32 batchsize);

  void InitParticle(const CuMatrix<BaseFloat> &vt) {
    fp_vt_.CopyFromMat(vt);

    cu_rand_clean_hid_.RandUniform(&fp_ha_);

    cu_rand_noise_hid_.RandUniform(&fp_hs_);

  }

  void InferBatchsizeChange(int32 batchsize);

  void TrainBatchsizeChange(int32 batchsize);

  void SetZMomentum(BaseFloat value) {
    z_momentum_ = value;
  }

  BaseFloat GetZMomentum(BaseFloat value) {
    return z_momentum_;
  }

  void SetZStartIter(int32 value) {
    z_start_iter_ = value;
  }

  int32 GetZStartIter() {
    return z_start_iter_;
  }

  void SetNumInferenceIters(int32 value) {
    num_infer_iters_ = value;
  }

  int32 GetNumInferenceIters() {
    return num_infer_iters_;
  }

  /*
   * Functions inhereted from RbmBase.
   */
  void WriteAsNnet(std::ostream& os, bool binary) const {
    KALDI_ERR<< "Not implemented for RoRbm!";
  }

  void WriteAsAutoEncoder(std::ostream& os, bool isEncoder, bool binary) const {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  void Reconstruct(const CuMatrix<BaseFloat> &hid_state, CuMatrix<BaseFloat> *vis_probs) {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ERR<< "Not implemented for RoRbm!";
  }

  virtual void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &err) {
    KALDI_ERR<< "Not implemented for RoRbm!";
  }
private:
  // Model parameters for the noisy input
  CuVector<BaseFloat> bt_;///< Input bias vector \tilde{b}
  CuVector<BaseFloat> lamt2_;///< Input variance vector 1.0/{\tilde{\sigma}^2}

  // Model parameters for the gating function
  CuVector<BaseFloat> gamma2_;///< Gating vector gamma_square

  // Model parameters for the clean GRBM
  CuMatrix<BaseFloat> clean_vis_hid_;///< Matrix with neuron weights, size [clean_hid_dim_, vis_dim_]
  CuVector<BaseFloat> clean_vis_bias_;///< Vector with biases
  CuVector<BaseFloat> clean_hid_bias_;///< Vector with biases
  CuVector<BaseFloat> clean_vis_std_;///< Standard deviation of the clean GRM inputs, \sigma
  CuVector<BaseFloat> clean_vis_var_;

  // Model parameters for the noise indicator RBM
  CuMatrix<BaseFloat> U_;// noise_vis_hid_; ///< Weight matrix U, size [noise_hid_dim_, vis_dim_]
  CuVector<BaseFloat> d_;// noise_vis_bias_; ///< Visible bias vector d
  CuVector<BaseFloat> e_;// noise_hid_bias_ori_; ///< Hidden bias vector e, This is model parameter

  // Variables for parameter updates
  CuMatrix<BaseFloat> U_corr_;///< Matrix for noise RBM weight updates
  CuVector<BaseFloat> d_corr_;///< Vector for noise visible bias updates
  CuVector<BaseFloat> e_corr_;///< Vector for noise hidden bias updates
  CuVector<BaseFloat> bt_corr_;///< Vector for input bias updates
  CuVector<BaseFloat> lamt2_corr_;///< Vector for input variance updates
  CuVector<BaseFloat> gamma2_corr_;///< Vector for gamm2 updates

  CuMatrix<BaseFloat> U_pos_, U_neg_;
  CuVector<BaseFloat> d_pos_, d_neg_;
  CuVector<BaseFloat> e_pos_, e_neg_;
  CuVector<BaseFloat> bt_pos_, bt_neg_;// visible node bias
  CuVector<BaseFloat> lamt2_pos_, lamt2_neg_;
  CuVector<BaseFloat> gamma2_pos_, gamma2_neg_;

  // intermediate variables
  /* size: [n, vis_dim_/clean_hid_dim_/noise_hid_dim_]
   * fp_*: fantasy particles (needed for negative phase of SAP)
   *
   */
  CuMatrix<BaseFloat> fp_v_, fp_vt_, v_sample_;
  CuMatrix<BaseFloat> ha_, haprob_, fp_ha_;  /// for clean RBM hidden activations
  CuMatrix<BaseFloat> hs_, hsprob_, fp_hs_;/// for noise RBM hidden activations
  CuMatrix<BaseFloat> mu_, mu_hat_, mu_t_hat_;
  CuMatrix<BaseFloat> s_, phi_s_, fp_s_, z_;
  CuMatrix<BaseFloat> log_sprob_0_, log_sprob_1_;
  CuMatrix<BaseFloat> mat_tmp_;
  CuMatrix<BaseFloat> v_condstd_, fp_vt_condstd_;
  CuMatrix<BaseFloat> v_condmean_, fp_vt_condmean_;

  /* size: [noise_hid_dim_, vis_dim_] */
  CuMatrix<BaseFloat> U_tmp_;

  /* size: [1, vis_dim_]*/
  CuVector<BaseFloat> std_hat_;
  CuVector<BaseFloat> inv_gamma2_tmp_;
  CuVector<BaseFloat> vec_tmp_, vec_tmp2_;
  //CuVector<BaseFloat> s_mu_;
  CuVector<BaseFloat> lamt2_hat_;

  RbmNodeType vis_type_;
  RbmNodeType clean_hid_type_;
  RbmNodeType noise_hid_type_;

  int32 vis_dim_;///< visible layer dim, same for \tilde{v}, v and s
  int32 clean_hid_dim_;///< hidden layer dim for clean GRBM
  int32 noise_hid_dim_;///< hidden layer dim for noise indicator RBM

  /* As the seeding is done in the host and it is slow, it's better to use
   * different random generators for matrices with different sizes.
   */
  CuRand<BaseFloat> cu_rand_vis_;  ///< random generator for visible data
  CuRand<BaseFloat> cu_rand_clean_hid_;///< random generator for clean hidden data
  CuRand<BaseFloat> cu_rand_noise_hid_;///< random generator for noise hidden data

  BaseFloat z_momentum_;
  int32 z_start_iter_;

  int32 num_infer_iters_;

  int32 batch_size_;

  bool first_data_bunch_;

};

}  // namespace

#endif /* KALDI_NNET_RORBM_H_ */
