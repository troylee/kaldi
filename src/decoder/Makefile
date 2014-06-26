all:

EXTRA_CXXFLAGS = -Wno-sign-compare -O3
include ../kaldi.mk

TESTFILES = 

OBJFILES = decodable-am-diag-gmm.o training-graph-compiler.o decodable-am-sgmm.o \
   decodable-am-tied-diag-gmm.o decodable-am-tied-full-gmm.o \
   lattice-simple-decoder.o lattice-faster-decoder.o faster-decoder.o \
   decodable-am-sgmm2.o lattice-tracking-decoder.o

LIBNAME = kaldi-decoder

ADDLIBS = ../transform/kaldi-transform.a ../tree/kaldi-tree.a \
     ../sgmm/kaldi-sgmm.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../util/kaldi-util.a \
     ../base/kaldi-base.a ../matrix/kaldi-matrix.a 

include ../makefiles/default_rules.mk


