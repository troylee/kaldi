all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I ../../tools/portaudio/install/include

UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
    EXTRA_LDLIBS = ../../tools/portaudio/install/lib/libportaudio.a
ifneq ($(wildcard ../../tools/portaudio/install/include/pa_linux_alsa.h),)
    EXTRA_LDLIBS += -lasound
else
    EXTRA_LDLIBS += -lrt
endif
else
    EXTRA_LDLIBS = -L ../../tools/portaudio/install/lib/ -lportaudio
endif

include ../kaldi.mk

BINFILES = online-net-client online-server-gmm-decode-faster online-gmm-decode-faster online-wav-gmm-decode-faster

OBJFILES =

LIBFILE =

ADDLIBS = ../online/kaldi-online.a ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a  ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk


