/*
 * vts-model-decode.cc
 *
 *  Created on: Oct 20, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Estimate noise using VTS on each utterance and then decoding with
 *  the compensated models.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "decoder/decodable-am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "util/timer.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc
#include "gmm/diag-gmm-normal.h"
#include "vts/vts-first-order.h"

namespace kaldi {

fst::Fst<fst::StdArc> *ReadNetwork(std::string filename) {
  // read decoding network FST
  Input ki(filename);  // use ki.Stream() instead of is.
  if (!ki.Stream().good())
    KALDI_ERR << "Could not open decoding-graph FST " << filename;

  fst::FstHeader hdr;
  if (!hdr.Read(ki.Stream(), "<unknown>")) {
    KALDI_ERR << "Reading FST: error reading FST header.";
  }
  if (hdr.ArcType() != fst::StdArc::Type()) {
    KALDI_ERR << "FST with arc type " << hdr.ArcType() << " not supported.\n";
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst < fst::StdArc > *decode_fst = NULL;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst < fst::StdArc > ::Read(ki.Stream(), ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst < fst::StdArc > ::Read(ki.Stream(), ropts);
  } else {
    KALDI_ERR << "Reading FST: unsupported FST type: " << hdr.FstType();
  }
  if (decode_fst == NULL) {  // fst code will warn.
    KALDI_ERR << "Error reading FST (after reading header).";
    return NULL;
  } else {
    return decode_fst;
  }
}

}  // Kalid namespace

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Online 1st-order VTS model compensation and decoding using Diagonal GMM-based model.\n"
            "Usage:  vts-model-decode [options] model-in fst-in features-rspecifier words-wspecifier noiseparams-wspecifier [alignments-wspecifier [lattice-wspecifier]]\n"
            "Note: lattices, if output, will just be linear sequences. Features are MFCC_0_D_A, C0 is the last item.\n";
    ParseOptions po(usage);
    bool allow_partial = true;
    BaseFloat acoustic_scale = 0.1;
    int32 noise_frames = 20;
    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;
    int32 em_iterations = 2;
    int32 noise_iterations = 4;
    BaseFloat variance_lrate = 0.1;
    BaseFloat max_noise_mean_magnitude = 999999.0;

    std::string word_syms_filename;
    FasterDecoderOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("noise-frames", &noise_frames,
                "Number of frames at the begining and ending of"
                " each sentence used for noise estimation");
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");
    po.Register("em-iterations", &em_iterations,
                "Number of iterations for EM.");
    po.Register("noise-iterations", &noise_iterations,
                "Number of iterations for noise parameter estimation in one EM step");
    po.Register("variance-lrate", &variance_lrate,
                "Learning rate for additive noise diagonal covariance");
    po.Register("max-noise-mean-magnitude", &max_noise_mean_magnitude,
                "Maximum magnitude value for the noise means (including"
                " both convolutional and additive noises)");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "Produce output even when final state was not reached");
    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        words_wspecifier = po.GetArg(4),
        noiseparams_wspecifier = po.GetArg(5),
        alignment_wspecifier = po.GetOptArg(6),
        lattice_wspecifier = po.GetOptArg(7);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    CompactLatticeWriter clat_writer(lattice_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
        KALDI_ERR << "Could not read symbol table from file "
            << word_syms_filename;
      }
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    DoubleVectorWriter noiseparams_writer(noiseparams_wspecifier);

    // It's important that we initialize decode_fst after feature_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    fst::Fst < fst::StdArc > *decode_fst = ReadNetwork(fst_rxfilename);

    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    FasterDecoder decoder(*decode_fst, decoder_opts);

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    Timer timer;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      Matrix<BaseFloat> features(feature_reader.Value());
      feature_reader.FreeCurrent();

      KALDI_VLOG(1) << "Current utterance: " << key;

      if (features.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key;
        num_fail++;
        continue;
      }

      if (features.NumRows() < noise_frames * 2) {
        KALDI_WARN << "Too short utterance for VTS: " << key << " ("
            << features.NumRows() << " frames)";
      }

      int32 feat_dim = features.NumCols();
      if (feat_dim != 3 * num_cepstral) {
        KALDI_ERR << "Could not decode the features, "
            << 3 * num_cepstral <<"D MFCC_0_D_A is expected!";
      }

      /************************************************
       Necessary parameters for VTS compensation
       *************************************************/

      // noise model parameters, current estimate and last estimate
      Vector<double> mu_h(feat_dim), mu_z(feat_dim), var_z(feat_dim),
          mu_h0(feat_dim), mu_z0(feat_dim), var_z0(feat_dim);

      // saved parameters for noise model computation
      std::vector<Matrix<double> > Jx(am_gmm.NumGauss()), Jz(am_gmm.NumGauss());
      Vector<double> gamma(am_gmm.NumGauss());  // sum( gamma_t_m )
      Matrix<double> gamma_p(am_gmm.NumGauss(), feat_dim);  // sum( gamma_t_m * y_t )
      Matrix<double> gamma_q(am_gmm.NumGauss(), feat_dim);  // sum( gamma_t_m * y_t * y_t )

      /************************************************
       Estimate the initial noise parameters
       *************************************************/

      EstimateInitialNoiseModel(features, feat_dim, num_cepstral,
                                noise_frames, true, &mu_h, &mu_z,
                                &var_z);

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Initial Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Initial Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Initial Convoluational Noise Mean: " << mu_h;
      }

      /************************************************
       Compensate the model
       *************************************************/

      // model after compensation
      AmDiagGmm noise_am_gmm;
      // Initialize with the clean speech model
      noise_am_gmm.CopyFromAmDiagGmm(am_gmm);

      CompensateModel(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat,
                      inv_dct_mat, noise_am_gmm, Jx, Jz);

      if (g_kaldi_verbose_level >= 20) {
        // verify the model is indeed changed
        {
          Output ko("noise_model.txt", false);
          trans_model.Write(ko.Stream(), false);
          noise_am_gmm.Write(ko.Stream(), false);
        }
      }

      /************************************************
       EM iterative estimation of the noise model
       *************************************************/
      bool em_continue = true; // whether to finish all the EM iterations
      for (int32 em_iter = 0; em_continue && em_iter < em_iterations; ++em_iter) {

        KALDI_LOG << "EM Iteration " << em_iter;

        /*
         * Generate the hypotheses based on current model and
         * get the sufficient statistics.
         */
        // Decode with the compensated noisy speech model
        DecodableAmDiagGmmScaled gmm_decodable(noise_am_gmm, trans_model,
                                               features, acoustic_scale);
        decoder.Decode(&gmm_decodable);

        fst::VectorFst<LatticeArc> decoded;  // linear FST.
        decoder.GetBestPath(&decoded);

        if (!decoder.ReachedFinal()) {
          KALDI_ERR << "Decoder did not reach end-state, "
              << "noise model estimation is stopped!";
        }

        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        BaseFloat like = -weight.Value1() - weight.Value2();

        KALDI_LOG << "Loglike from decoding: " << like;
        KALDI_LOG << "Alignment size: " << alignment.size()
            << ", # of features: " << features.NumRows();

        KALDI_ASSERT((alignment.size() == features.NumRows()));

        for (int32 noise_iter = 0; noise_iter < noise_iterations;
            ++noise_iter) {

          /*
           * Convert alignment to state posterior, which will be used for Gaussian posterior
           * computation.
           */

          BaseFloat tot_like_this_file = AccumulatePosteriorStatistics(
              noise_am_gmm, trans_model, alignment, features, gamma, gamma_p,
              gamma_q);

          KALDI_LOG << "Loglike from accumulation: " << tot_like_this_file;

          KALDI_LOG << "Noise estimation iteration " << noise_iter;

          // keep a copy of the old parameters
          mu_h0.CopyFromVec(mu_h);
          mu_z0.CopyFromVec(mu_z);
          var_z0.CopyFromVec(var_z);

          bool new_mean_estimate = false;
          bool new_var_estimate = false;

          /*
           * Estimate noise means, mu_h, mu_z (only have static coefficients)
           */
          SubVector<double> mu_h_s(mu_h, 0, num_cepstral), mu_z_s(mu_z, 0,
                                                                  num_cepstral);
          EstimateStaticNoiseMean(noise_am_gmm, gamma, gamma_p, gamma_q, Jx, Jz,
                                  num_cepstral, max_noise_mean_magnitude, mu_h_s, mu_z_s);

          if (g_kaldi_verbose_level >= 1) {
            KALDI_LOG << "New Additive Noise Mean: " << mu_z;
            KALDI_LOG << "New Convoluational Noise Mean: " << mu_h;
          }

          new_mean_estimate = BackOff(am_gmm, trans_model, alignment, features,
                                      num_cepstral, num_fbank, dct_mat,
                                      inv_dct_mat, mu_h0, mu_z0, var_z0, mu_h,
                                      true, mu_z, true, var_z, false,
                                      noise_am_gmm, Jx, Jz);

          if (fabs(variance_lrate) > 1e-6) {
            /*
             * Estimate noise variance, have static, delta and accelerate parameters.
             * if variance_lrate == 0, then no variance estimation.
             */
            EstimateAdditiveNoiseVariance(noise_am_gmm, gamma, gamma_p, gamma_q,
                                          Jz, num_cepstral, feat_dim,
                                          variance_lrate, var_z);

            new_var_estimate = BackOff(am_gmm, trans_model, alignment, features,
                                       num_cepstral, num_fbank, dct_mat,
                                       inv_dct_mat, mu_h0, mu_z0, var_z0, mu_h,
                                       false, mu_z, false, var_z, true,
                                       noise_am_gmm, Jx, Jz);
          }

          // If the estimatation completely revert back to the previous estimation,
          // no need to do more iterations.
          if (!new_mean_estimate && !new_var_estimate){
            KALDI_LOG << "No updates are acceptable, stopping ... ";
            em_continue=false;
            break;
          }

        }

      }

      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG << "Final Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Final Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Final Convoluational Noise Mean: " << mu_h;
      }

      // Writting the noise parameters out
      noiseparams_writer.Write(key + "_mu_h", mu_h);
      noiseparams_writer.Write(key + "_mu_z", mu_z);
      noiseparams_writer.Write(key + "_var_z", var_z);

      KALDI_LOG << "Final decoding starting ... ";

      // Decode with the compensated noisy speech model
      DecodableAmDiagGmmScaled gmm_decodable(noise_am_gmm, trans_model,
                                             features, acoustic_scale);
      decoder.Decode(&gmm_decodable);

      fst::VectorFst<LatticeArc> decoded;  // linear FST.

      if ((allow_partial || decoder.ReachedFinal())
          && decoder.GetBestPath(&decoded)) {
        if (!decoder.ReachedFinal())
          KALDI_WARN << "Decoder did not reach end-state, "
              << "outputting partial traceback since --allow-partial=true";
        num_success++;
        if (!decoder.ReachedFinal())
          KALDI_WARN
              << "Decoder did not reach end-state, outputting partial traceback.";
        std::vector<int32> alignment;
        std::vector<int32> words;
        LatticeWeight weight;
        frame_count += features.NumRows();

        GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

        words_writer.Write(key, words);
        if (alignment_writer.IsOpen()) {
          alignment_writer.Write(key, alignment);
        }

        if (lattice_wspecifier != "") {
          if (acoustic_scale != 0.0) {  // We'll write the lattice without acoustic scaling
            fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale),
                              &decoded);
          }
          fst::VectorFst<CompactLatticeArc> clat;
          ConvertLattice(decoded, &clat, true);
          clat_writer.Write(key, clat);
        }

        if (word_syms != NULL) {
          std::cerr << key << ' ';
          for (size_t i = 0; i < words.size(); i++) {
            std::string s = word_syms->Find(words[i]);
            if (s == "") {
              KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
            }
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }
        BaseFloat like = -weight.Value1() - weight.Value2();
        tot_like += like;
        KALDI_LOG << "Log-like per frame for utterance " << key << " is "
            << (like / features.NumRows()) << " over " << features.NumRows()
            << " frames.";
        KALDI_VLOG(2) << "Cost for utterance " << key << " is "
            << weight.Value1() << " + " << weight.Value2();
      } else {
        num_fail++;
        KALDI_WARN << "Did not successfully decode utterance " << key
            << ", len = " << features.NumRows();
      }
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] " << elapsed
        << "s: real-time factor assuming 100 frames/sec is "
        << (elapsed * 100.0 / frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
        << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is "
        << (tot_like / frame_count) << " over " << frame_count << " frames.";

    if (word_syms)
      delete word_syms;
    delete decode_fst;
    return (num_success != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

