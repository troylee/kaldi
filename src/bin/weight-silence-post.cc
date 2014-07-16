// bin/weight-silence-post.cc

// Copyright 2009-2011  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

void WeightSilencePost(const TransitionModel &trans_model,
                       const ConstIntegerSet<int32> &silence_set,
                       BaseFloat silence_scale,
                       Posterior *post) {
  for (size_t i = 0; i < post->size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 tid = (*post)[i][j].first,
          phone = trans_model.TransitionIdToPhone(tid);
      BaseFloat weight = (*post)[i][j].second;
      if (silence_set.count(phone) != 0) {  // is a silence.
        if (silence_scale != 0.0)
          this_post.push_back(std::make_pair(tid, weight*silence_scale));
      } else {
        this_post.push_back(std::make_pair(tid, weight));
      }
    }
    (*post)[i].swap(this_post);
  }
}


void WeightSilencePostDistributed(const TransitionModel &trans_model,
                                  const ConstIntegerSet<int32> &silence_set,
                                  BaseFloat silence_scale,
                                  Posterior *post) {
  for (size_t i = 0; i < post->size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());
    BaseFloat sil_weight = 0.0, nonsil_weight = 0.0;
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 tid = (*post)[i][j].first,
          phone = trans_model.TransitionIdToPhone(tid);
      BaseFloat weight = (*post)[i][j].second;
      if (silence_set.count(phone) != 0) sil_weight += weight;
      else nonsil_weight += weight;
    }
    KALDI_ASSERT(sil_weight >= 0.0 && nonsil_weight >= 0.0); // This "distributed"
    // weighting approach doesn't make sense if we have negative weights.
    if (sil_weight + nonsil_weight == 0.0) continue;
    BaseFloat frame_scale = (sil_weight * silence_scale + nonsil_weight) /
                            (sil_weight + nonsil_weight);
    if (frame_scale != 0.0) {
      for (size_t j = 0; j < (*post)[i].size(); j++) {
        int32 tid = (*post)[i][j].first;
        BaseFloat weight = (*post)[i][j].second;
        this_post.push_back(std::make_pair(tid, weight * frame_scale));
      }
    }
    (*post)[i].swap(this_post);
  }
}

} // end namespace

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Apply weight to silences in posteriors\n"
        "Usage:  weight-silence-post [options] <silence-weight> <silence-phones> "
        "<model> <posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " weight-silence-post 0.0 1:2:3 1.mdl ark:1.post ark:nosil.post\n";

    ParseOptions po(usage);

    bool distribute = false;

    po.Register("distribute", &distribute, "If true, rather than weighting the "
                "individual posteriors, apply the weighting to the whole frame: "
                "i.e. on time t, scale all posterior entries by "
                "p(sil)*silence-weight + p(non-sil)*1.0");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string silence_weight_str = po.GetArg(1),
        silence_phones_str = po.GetArg(2),
        model_rxfilename = po.GetArg(3),
        posteriors_rspecifier = po.GetArg(4),
        posteriors_wspecifier = po.GetArg(5);

    BaseFloat silence_weight = 0.0;
    if (!ConvertStringToReal(silence_weight_str, &silence_weight))
      KALDI_ERR << "Invalid silence-weight parameter: expected float, got \""
                 << silence_weight_str << '"';
    std::vector<int32> silence_phones;
    if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    if (silence_phones.empty())
      KALDI_WARN <<"No silence phones, this will have no effect";
    ConstIntegerSet<int32> silence_set(silence_phones);  // faster lookup.

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    int32 num_posteriors = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      num_posteriors++;
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post = posterior_reader.Value();

      if( distribute ) {
        WeightSilencePostDistributed(trans_model, silence_set, silence_weight, &post);
      } else {
        WeightSilencePost(trans_model, silence_set, silence_weight, &post);
      }

      posterior_writer.Write(posterior_reader.Key(), post);
    }
    KALDI_LOG << "Done " << num_posteriors << " posteriors.";
    return (num_posteriors != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


