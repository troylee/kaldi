// nnetbin/codevec-train-xent-hardlab-perutt.cc

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-codebl.h"

#include <algorithm>
#include <ctime>

int main(int argc, char *argv[]) {

  try {

    std::srand( unsigned(std::time(0)));

    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Learn code vector only by stochastic gradient descent.\n"
            "Usage:  codevec-train-xent-hardlab-frmshuff [options] <adapt-model-in> <backend-mode-in> <feature-rspecifier> "
            "<alignments-rspecifier> <set2utt-map> <codevec-rspecifier> [<codevec-wspecifier>]\n"
            "e.g.: \n"
            " codevec-train-xent-hardlab-perutt nnet.init backend.nnet scp:train.scp ark:train.ali ark:set2utt.map"
            " ark:codes.ark ark:new_code.ark\n";

    ParseOptions po(usage);
    bool binary = false,
        crossvalidate = false,
        randomize = true,
        shuffle = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");
    po.Register("shuffle", &shuffle,
                "Perform the utterance-level shuffling during training");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad,
                "Whether to average the gradient in the bunch");

    std::string learn_factors = "";
    po.Register(
        "learn-factors",
        &learn_factors,
        "Learning factor for each updatable layer, separated by ',' and work together with learn rate");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    po.Read(argc, argv);

    if (po.NumArgs() != 7 - (crossvalidate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        backend_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignments_rspecifier = po.GetArg(4),
        set2utt_rspecifier = po.GetArg(5),
        codevec_rspecifier = po.GetArg(6);

    std::string codevec_wspecifier;
    if (!crossvalidate) {
      codevec_wspecifier = po.GetArg(7);
    }

    Nnet nnet_transf, nnet_backend;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    nnet_backend.Read(backend_model_filename);

    Nnet nnet;
    nnet.Read(model_filename);

    if (learn_factors == "") {
      nnet.SetLearnRate(learn_rate, NULL);
    } else {
      nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    }
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);
    nnet.SetAverageGrad(average_grad);

    /*
     * Find out all the <codebl> layers.
     */
    int32 num_codebl = 0, codevec_dim = 0;
    std::vector<CodeBL*> layers_codebl;
    for (int32 li = 0; li < nnet.LayerCount(); ++li) {
      if (nnet.Layer(li)->GetType() == Component::kCodeBL) {
        layers_codebl.push_back(static_cast<CodeBL*>(nnet.Layer(li)));
        ++num_codebl;
        if (codevec_dim == 0){
          codevec_dim = layers_codebl[num_codebl-1]->GetCodeVecDim();
        } else if (codevec_dim != layers_codebl[num_codebl-1]->GetCodeVecDim()) {
          KALDI_ERR << "Inconsistent code vector dimension for different <codebl> layers!";
        }
        // disable weight update for <codebl> layers
        layers_codebl[num_codebl-1]->ConfigWeightUpdate(false);
      }
    }
    KALDI_LOG << "Totally " << num_codebl << " among " << nnet.LayerCount()
        << " layers of the nnet are <codebl> layers.";

    kaldi::int64 tot_t = 0;

    SequentialTokenVectorReader set2utt_reader(set2utt_rspecifier);
    RandomAccessBaseFloatVectorReader codevec_reader(codevec_rspecifier);

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    BaseFloatVectorWriter codevec_writer(codevec_wspecifier);

    Cache cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    Vector<BaseFloat> host_codevec;

    CuVector<BaseFloat> codevec_cur(codevec_dim), codevec_corr(codevec_dim);
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, glob_err, backend_out, backend_err, nnet_err;
    std::vector<int32> targets;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0,
        num_cache = 0, num_set = 0;
    for (; !set2utt_reader.Done(); set2utt_reader.Next()) {
      std::string setkey = set2utt_reader.Key();
      codevec_cur.CopyFromVec(codevec_reader.Value(setkey));

      std::cerr << "Set # " << ++num_set << " - " << setkey << " :\n";

      // Update the nnet's codebl layers with this new code vector
      for (int32 li=0; li < num_codebl; ++li) {
        (layers_codebl[li])->SetCodeVec(codevec_cur);
      }

      // all the utts belong to this set
      std::vector<std::string> uttlst(set2utt_reader.Value());
      if(shuffle){
        std::random_shuffle(uttlst.begin(), uttlst.end());
      }

      for (int32 uid = 0; uid < uttlst.size();) {

        // fill the cache
        while (!cache.Full() && uid < uttlst.size()) {
          std::string key = uttlst[uid];
          if (!alignments_reader.HasKey(key)) {
            num_no_alignment++;
          } else {
            // get feature alignment pair
            const Matrix<BaseFloat> &mat = feature_reader.Value(key);
            const std::vector<int32> &alignment = alignments_reader.Value(key);
            // chech for dimension
            if ((int32) alignment.size() != mat.NumRows()) {
              KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
              num_other_error++;
              continue;
            }
              // push features to GPU
            feats.CopyFromMat(mat);
            // possibly apply transform
            nnet_transf.Feedforward(feats, &feats_transf);
            // add to cache
            cache.AddData(feats_transf, alignment);
            num_done++;
          }
          Timer t_features;
          uid += 1; // next utterance
          time_next += t_features.Elapsed();
        }
        // randomize
        if (!crossvalidate && randomize) {
          cache.Randomize();
        }
        // report
        std::cerr << "Cache #" << ++num_cache << " "
            << (cache.Randomized() ? "[RND]" : "[NO-RND]")
            << " segments: " << num_done
            << " frames: " << tot_t << "\n";
        // train with the cache
        while (!cache.Empty()) {
          // get block of feature/target pairs
          cache.GetBunch(&nnet_in, &targets);
          // train
          nnet.Propagate(nnet_in, &nnet_out);
          nnet_backend.Propagate(nnet_out, &backend_out);

          xent.EvalVec(backend_out, targets, &glob_err);
          if (!crossvalidate) {
            nnet_backend.Backpropagate(glob_err, &backend_err);
            nnet.Backpropagate(backend_err, &nnet_err); // to compute the code error, we need to propagete through 1st layer

            // accumulate code vector correction
            codevec_corr.SetZero();
            for(int32 li = 0; li < num_codebl; ++li){
              codevec_corr.AddVec(1.0, layers_codebl[li]->GetCodeVecCorr(), 1.0);
            }
            // update the current code vector with the average correction
            codevec_cur.AddVec(1.0/num_codebl, codevec_corr, 1.0);
            for(int32 li = 0; li < num_codebl; ++li) {
              layers_codebl[li]->SetCodeVec(codevec_cur);
            }
          }
          tot_t += nnet_in.NumRows();
        } // end while cache

      } // end for uttlst

      // Save the new code vector
      if (!crossvalidate){
        codevec_cur.CopyToVec(&host_codevec);
        codevec_writer.Write(setkey, host_codevec);
      }

    } // end for uttset

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_set << " sets.";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  } // end try
} // end main
