// nnet/nnet-cache-xent-tgtmat.h

/*
 * Created on: Sep 15, 2013
 *     Author: Troy Lee (troy.lee2008@gmail.com)
 *
 */

#ifndef KALDI_NNET_CACHE_H
#define KALDI_NNET_CACHE_H

#include "base/kaldi-math.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

/**
 * The feature-target pair cache
 */
class CacheXentTgtMat {
  typedef enum { EMPTY, FILLING, FULL, EMPTYING } State;

 public:
  CacheXentTgtMat() : state_(EMPTY), filling_pos_(0), emptying_pos_(0),
            cachesize_(0), bunchsize_(0), randomized_(false) 
  { }
  ~CacheXentTgtMat() { }
 
  /// Initialize the cache
  void Init(int32 cachesize, int32 bunchsize);

  /// Add data to cache
  void AddData(const CuMatrix<BaseFloat> &features, const std::vector<int32> &aligns, const CuMatrix<BaseFloat> &targets);
  /// Randomizes the cache
  void Randomize();
  /// Get the bunch of training data from cache
  void GetBunch(CuMatrix<BaseFloat> *features, std::vector<int32> *aligns, CuMatrix<BaseFloat> *targets);


  /// Returns true if the cache was completely filled
  bool Full() { 
    return (state_ == FULL); 
  }
  
  /// Returns true if the cache is empty
  bool Empty() { 
    return (state_ == EMPTY || filling_pos_ < bunchsize_); 
  }

  /// Returns true if the cache is empty
  bool Randomized() { 
    return randomized_;
  }


 private:
  struct GenerateRandom {
    int32 operator()(int32 max) { 
      // return lrand48() % max; 
      return RandInt(0, max-1); 
    }
  };

  State state_; ///< Current state of the cache

  size_t filling_pos_;  ///< Number of frames filled to cache by AddData
  size_t emptying_pos_; ///< Number of frames given by cache by GetBunch
  
  size_t cachesize_; ///< Size of cache
  size_t bunchsize_; ///< Size of bunch

  bool randomized_;

  CuMatrix<BaseFloat> features_; ///< Feature cache
  CuMatrix<BaseFloat> features_random_; ///< Feature cache
  CuMatrix<BaseFloat> features_leftover_; ///< Feature cache
  
  std::vector<int32> aligns_; ///< Xent target vector cache
  std::vector<int32> aligns_random_; ///< Xent target vector cache
  std::vector<int32> aligns_leftover_; ///< Xent target vector cache

  CuMatrix<BaseFloat> targets_;  ///< Desired vector cache
  CuMatrix<BaseFloat> targets_random_;  ///< Desired vector cache
  CuMatrix<BaseFloat> targets_leftover_;  ///< Desired vector cache

  std::vector<int32> randmask_;
  CuStlVector<int32> randmask_device_;

}; 
 
  
} // namespace

#endif
