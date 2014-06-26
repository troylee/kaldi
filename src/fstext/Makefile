
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = determinize-star-test \
      pre-determinize-test trivial-factor-weight-test \
      context-fst-test factor-test table-matcher-test fstext-utils-test \
      remove-eps-local-test rescale-test lattice-weight-test  \
      determinize-lattice-test lattice-utils-test deterministic-fst-test \
      determinize-lattice-pruned-test 

OBJFILES =  

# actually, this library is currently empty.  Everything is a header.
LIBNAME = 

# tree and matrix archives needed for test-context-fst
ADDLIBS =  ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../util/kaldi-util.a

include ../makefiles/default_rules.mk


