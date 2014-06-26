all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

TESTFILES = am-sgmm-test estimate-am-sgmm-test  \
   fmllr-sgmm-test

OBJFILES = am-sgmm.o estimate-am-sgmm.o estimate-am-sgmm-ebw.o fmllr-sgmm.o \
          am-sgmm-project.o

LIBNAME = kaldi-sgmm

ADDLIBS = ../base/kaldi-base.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
	../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../transform/kaldi-transform.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk


