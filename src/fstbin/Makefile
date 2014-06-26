
# make "all" the target.
all:

# Disable linking math libs because  not needed here.  Just for compilation speed.
# no, it's now needed for context-fst-test.
# MATHLIB = NONE

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

TESTFILES = 

OBJFILES = 

BINFILES = fstdeterminizestar  \
           fstrmsymbols fstisstochastic fstminimizeencoded fstmakecontextfst \
           fstmakecontextsyms fstaddsubsequentialloop fstaddselfloops  \
           fstrmepslocal fstcomposecontext fsttablecompose fstrand fstfactor \
           fstdeterminizelog fstphicompose fstpropfinal fstcopy

# actually, this library is currently empty.  Everything is a header.
LIBFILE = 

ADDLIBS = ../base/kaldi-base.a ../util/kaldi-util.a

include ../makefiles/default_rules.mk

