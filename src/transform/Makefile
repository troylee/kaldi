all:

include ../kaldi.mk

TESTFILES = regtree-fmllr-diag-gmm-test lda-estimate-test \
      regression-tree-test fmllr-diag-gmm-test exponential-transform-test \
      regtree-mllr-diag-gmm-test fmpe-test

OBJFILES = regression-tree.o regtree-mllr-diag-gmm.o lda-estimate.o \
    regtree-fmllr-diag-gmm.o cmvn.o transform-common.o fmllr-diag-gmm.o \
    lvtln.o mllt.o exponential-transform.o hlda.o fmpe.o basis-fmllr-diag-gmm.o \
    compressed-transform-stats.o


LIBNAME = kaldi-transform

ADDLIBS = ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

