/*
 * vts-first-order.h
 *
 *  Created on: Oct 20, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef KALDI_VTS_VTS_FIRST_ORDER_H_
#define KALDI_VTS_VTS_FIRST_ORDER_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"


namespace kaldi {

/*
 * Due to the feature normalization required for nnet,
 * we have to convert the GMMs estimated from the normalized features
 * back to the original features;
 * and also from the original features to the normalized ones.
 */

// in-place change from original feature GMM to normalized feature GMM
void GmmToNormalizedGmm(const Vector<double> &mean, const Vector<double> &std,
                        AmDiagGmm &am_gmm);

// in-place change from normalized feature GMM to original feature GMM
void NormalizedGmmToGmm(const Vector<double> &mean, const Vector<double> &std,
                        AmDiagGmm &am_gmm);

/*
 * Compute the KL-divergence of two diagonal Gaussians.
 * Refer to: http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
 *
 * P represents the true distribution
 * Q represents the estimated distribution.
 *
 */
double KLDivergenceDiagGaussian(const Vector<double> &p_mean,
                                const Vector<double> &p_var,
                                const Vector<double> &q_mean,
                                const Vector<double> &q_var);

/*
 * Generate the DCT and inverse DCT transforms with/without the Cepstral liftering.
 */
void GenerateDCTmatrix(int32 num_cepstral, int32 num_fbank, BaseFloat ceplifter,
                       Matrix<double> *dct_mat,
                       Matrix<double> *inv_dct_mat);

/*
 * Get the global index of the first Gaussian of a given pdf.
 */
int32 GetGaussianOffset(const AmDiagGmm &am_gmm, int32 pdf_id);

/*
 * Dummy Noise Model estimation.
 *
 * Using the starting and ending frames to estimate the mean and variance of the
 * additive noise and the convolution noise is set to 0.
 *
 * noise_frames:  # of frames for estimation;
 *
 */
void EstimateInitialNoiseModel(const Matrix<BaseFloat> &features,
                               int32 feat_dim,
                               int32 num_static,
                               int32 noise_frames,
                               bool zero_mu_z_deltas,
                               Vector<double> *mu_h,
                               Vector<double> *mu_z,
                               Vector<double> *var_z);

/*
 * Compensate a single Diagonal Gaussian using estimated noise parameters.
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
                            const Vector<double> &var_z, int32 num_cepstral,
                            int32 num_fbank,
                            const Matrix<double> &dct_mat,
                            const Matrix<double> &inv_dct_mat,
                            Vector<double> &mean, Vector<double> &cov,
                            Matrix<double> &Jx,
                            Matrix<double> &Jz);

/*
 * Compensate a Diagonal Gaussian Mixture model.
 *
 * noise_gmm is inputed as clean GMM and compensated by this function
 *
 */
void CompensateDiagGmm(const Vector<double> &mu_h, const Vector<double> &mu_z,
                       const Vector<double> &var_z,
                       int32 num_cepstral,
                       int32 num_fbank,
                       const Matrix<double> &dct_mat,
                       const Matrix<double> &inv_dct_mat,
                       DiagGmm &noise_gmm,
                       std::vector<Matrix<double> > &Jx,
                       std::vector<Matrix<double> > &Jz);

/*
 * Do the compensation using the current noise model parameters.
 * Also keep the statistics of the Jx, and Jz for next iteration of noise estimation.
 *
 * noise_am_gmm is initialized as the clean model and after the function, it is the
 * compensated new model.
 *
 * Jx: the per-Gaussian component Jacobian of the mismatch function with respect to x,
 *      i.e. the clean speech mean;
 * Jz: the per-Gaussian component Jacobian of the mismatch function with respect to z,
 *      i.e. the additive noise mean;
 *
 *
 */
void CompensateModel(const Vector<double> &mu_h, const Vector<double> &mu_z,
                     const Vector<double> &var_z,
                     int32 num_cepstral,
                     int32 num_fbank,
                     const Matrix<double> &dct_mat,
                     const Matrix<double> &inv_dct_mat,
                     AmDiagGmm &noise_am_gmm,
                     std::vector<Matrix<double> > &Jx,
                     std::vector<Matrix<double> > &Jz);

/*
 * Compute the sufficient statistics for an utterance given the alignment.
 *
 * gamma: must be initialized to be zero vector, posterior of a Gaussian component,
 *        sum_t ( gamma_{t,s,m} );
 * gamma_p: must be initialized to be zero matrix,
 *        sum_t ( gamma_{t,s,m} * y_t );
 * gamma_q: must be initialized to be zero matrix,
 *        sum_t ( gamma_{t,s,m} * y_t * y_t );
 *
 * Return: the total log likelihood of the utterance ignoring
 * the transition probabilities.
 *
 */
BaseFloat AccumulatePosteriorStatistics(const AmDiagGmm &am_gmm,
                                        const TransitionModel &trans_model,
                                        const std::vector<int32> &alignment,
                                        const Matrix<BaseFloat> &features,
                                        Vector<double> &gamma,
                                        Matrix<double> &gamma_p,
                                        Matrix<double> &gamma_q);

/*
 * Compute the model likelihood of given feature and alignment.
 * The tranisition probabilities are ignored.
 *
 */
BaseFloat ComputeLogLikelihood(const AmDiagGmm &am_gmm,
                               const TransitionModel &trans_model,
                               const std::vector<int32> &alignment,
                               const Matrix<BaseFloat> &features);

/*
 * Back off if the new estimation doesn't increase the likelihood.
 *
 * The input noisy_am_gmm is the clean model compensated by mu_h0, mu_z0, var_z0.
 * After this function, it will be updated with the new noise estimation.
 *
 * If the estimation completely revert back to the original estimation, then return false;
 *
 *
 */
bool BackOff(const AmDiagGmm &clean_am_gmm, const TransitionModel &trans_model,
             const std::vector<int32> &alignment,
             const Matrix<BaseFloat> &features, int32 num_cepstral,
             int32 num_fbank,
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
             std::vector<Matrix<double> > &Jz);

/*
 * The AM_GMM is the noise compensated model using the existing noise estimation.
 *
 * The mu_h, mu_z and var_z contains the current estimation and will be updated in this function.
 *
 */
void EstimateStaticNoiseMean(const AmDiagGmm &noise_am_gmm,
                             const Vector<double> &gamma,
                             const Matrix<double> &gamma_p,
                             const Matrix<double> &gamma_q,
                             const std::vector<Matrix<double> > &Jx,
                             const std::vector<Matrix<double> > &Jz,
                             int32 num_cepstral, BaseFloat max_magnitude,
                             SubVector<double> &mu_h_s,
                             SubVector<double> &mu_z_s);

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
                                   Vector<double> &var_z);

/*
 * Multi-frame VTS first order compensation. Apply the VTS compensation to
 * multi-frame expanded GMM.
 *
 * noise_am_gmm are initialized as the clean model and after
 * this function will be the compensated models.
 *
 *
 */
void CompensateMultiFrameGmm(const Vector<double> &mu_h,
                             const Vector<double> &mu_z,
                             const Vector<double> &var_z, bool compensate_var,
                             int32 num_cepstral,
                             int32 num_fbank,
                             const Matrix<double> &dct_mat,
                             const Matrix<double> &inv_dct_mat,
                             int32 num_frames,
                             AmDiagGmm &noise_am_gmm);

/*
 * Noise VTS compensation for FBank features.
 */
void CompensateDiagGaussian_FBank(const Vector<double> &mu_h,
                                  const Vector<double> &mu_z,
                                  const Vector<double> &var_z, bool have_energy,
                                  int32 num_fbank,
                                  Vector<double> &mean, Vector<double> &cov,
                                  Matrix<double> &Jx,
                                  Matrix<double> &Jz);

}

#endif /* KALDI_VTS_VTS_FIRST_ORDER_H_ */
