
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = nnet-train-xent-hardlab-perutt nnet-forward rnnlm-train-perseq

OBJFILES =

TESTFILES =

ADDLIBS = ../nnet_cpu/kaldi-nnet.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk

