all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

TESTFILES = am-sgmm-test estimate-am-sgmm-test fmllr-sgmm-test

# removed estimate-am-sgmm-multi-test

OBJFILES = am-sgmm.o estimate-am-sgmm.o fmllr-sgmm.o sgmm-clusterable.o \
         estimate-am-sgmm-ebw.o estimate-am-sgmm-multi.o

LIBNAME = kaldi-sgmm

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a \
   ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk


