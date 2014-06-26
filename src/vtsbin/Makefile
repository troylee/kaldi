all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = compute-average-noise compute-utt-noise \
           feats-append-noise kl-divergence-gauss noise-mfc2fbk vts-acc-gmm-stats-ali \
           vts-align-compiled vts-mvn-global vts-mvn-global-fbank \
           vts-compute-likes vts-compute-mono-post vts-compute-obj \
           vts-est-global-noise vts-est-noise vts-feats vts-feats-fbank \
           vts-global-noise-decode vts-gmm-est vts-gmm-sum-accs \
           vts-init-global-noise vts-init-noise vts-model-decode \
           vts-noise-decode vts-sum-obj vts-noise-latgen
 
OBJFILES =

TESTFILES =

ADDLIBS = ../vts/kaldi-vts.a ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk



