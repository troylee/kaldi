/*
 * vts-acc-gmm-stats-ali.cc
 *
 *  Created on: Nov 23, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Accumulate statistics for re-estimating the pseudo-clean HMM model.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "util/timer.h"

#include "vts/vts-first-order.h"
#include "vts/vts-accum-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Accumulate sufficient statistics for updating the pseudo-clean GMM model.\n"
            "Usage:  vts-acc-gmm-stats-ali [options] model-in features-rspecifier "
            "alignments-rspecifier noise-rspecifier accs-stats-wspecifier\n"
            "Note: Features are MFCC_0_D_A in Kaldi format (i.e C0 is the first item).\n";
    ParseOptions po(usage);

    bool binary = true;
        po.Register("binary", &binary, "Write output in binary mode");

    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;

    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        noise_in_rspecifier = po.GetArg(4),
        accs_wxfilename = po.GetArg(5);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    VtsAccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, num_cepstral, kGmmAll);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_in_rspecifier);

    int num_success = 0, num_fail = 0;
    BaseFloat tot_like = 0.0;

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();

      KALDI_VLOG(1)<< "Current utterance: " << key;

      if (features.NumRows() == 0) {
        KALDI_WARN<< "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      int32 feat_dim = features.NumCols();
      if (feat_dim != 3 * num_cepstral) {
        KALDI_ERR<< "Could not decode the features, "
            << 3 * num_cepstral << "D MFCC_0_D_A is expected!";
      }

      /************************************************
       load alignment
       *************************************************/

      if (!alignments_reader.HasKey(key)) {
        KALDI_WARN<< "No alignment could be found for "
            << key << ", utterance ignored.";
        ++num_fail;
        continue;
      }
      const std::vector<int32> &alignment = alignments_reader.Value(key);

      if (alignment.size() != features.NumRows()) {
        KALDI_WARN<< "Alignments has wrong size " << (alignment.size())
            << " vs. " << (features.NumRows());
        ++num_fail;
        continue;
      }

      /************************************************
       load parameters for VTS compensation
       *************************************************/

      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_WARN<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        ++num_fail;
        continue;
      }

      Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
      Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
      Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));

      // saved parameters for noise model computation
      std::vector<Matrix<double> > Jx(am_gmm.NumGauss()), Jz(am_gmm.NumGauss());

      /************************************************
       Compensate the model
       *************************************************/

      // model after compensation
      AmDiagGmm noise_am_gmm;
      // Initialize with the clean speech model
      noise_am_gmm.CopyFromAmDiagGmm(am_gmm);

      CompensateModel(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat,
                      inv_dct_mat, noise_am_gmm, Jx, Jz);

      /*
       * Accumulate statistics from the alignment
       */
      BaseFloat tot_like_this_file = 0.0;
      for (size_t i = 0; i < alignment.size(); i++) {
        int32 tid = alignment[i],  // transition identifier.
            pdf_id = trans_model.TransitionIdToPdf(tid);
        trans_model.Accumulate(1.0, tid, &transition_accs);
        tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, noise_am_gmm,
                                                        Jx, features.Row(i),
                                                        pdf_id,
                                                        1.0);
      }

      tot_like += tot_like_this_file;
      ++num_success;

      if(num_success % 100 == 0){
        KALDI_LOG << "Done " << num_success << " files, log-like for " << key
            << " is " << tot_like_this_file << " over "
            << features.NumRows() << " frames.";
      }

    }

    KALDI_LOG<< "Done " << num_success << " utterances, failed for "
        << num_fail;
    KALDI_LOG<< "Overall log-likelihood per file is "
        << (tot_like / num_success) << " over " << num_success << " files.";

    {
      Output ko(accs_wxfilename, binary);
      transition_accs.Write(ko.Stream(), binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";

    return (num_success != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

