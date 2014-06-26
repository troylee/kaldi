
all:

include ../kaldi.mk

EXTRA_CXXFLAGS = -Wno-sign-compare

TESTFILES = kaldi-lattice-test

OBJFILES = kaldi-lattice.o lattice-functions.o word-align-lattice.o sausages.o

LIBNAME = kaldi-lat

ADDLIBS = ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk


