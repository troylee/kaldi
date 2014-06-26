

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

TESTFILES = #nnet-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o nnet-cache.o nnet-cache-tgtmat.o nnet-cache-xent-tgtmat.o nnet-posnegbl.o nnet-gaussbl.o nnet-rorbm.o

LIBNAME = kaldi-nnet

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a  ../util/kaldi-util.a 

include ../makefiles/default_rules.mk

