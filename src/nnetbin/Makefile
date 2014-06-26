
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)

BINFILES = nnet-train-xent-hardlab-perutt nnet-train-xent-hardlab-frmshuff nnet-train-mse-tgtmat-frmshuff \
		   nnet-forward nnet-copy rbm-train-cd1-frmshuff rbm-convert-to-nnet nnet-trim-n-last-transforms \
		   nnet-train-mmi-sequential dropoutbl-to-biasedlinearity train-noise-cmvn posnegbl-create \
		   posnegbl-forward posnegbl-train batched-feats gaussbl-create gaussbl-forward gaussbl-train \
		   rbms-convert-to-autoencoder compute-feats-mse weight-tied-ae-train append-lin sub-nnet \
		   rbm-to-maskedrbm copy-feats-to-text copy-feats-from-text rorbm-train-frmshuff grbm-train-frmshuff \
		   rbm-to-grbm rorbm-init rorbm-train-utt est-avg-mask-priors est-feat-masks interpolate-two-posts \
		   compute-mask-ratio est-feat-masks-with-pdf compute-posts-kl analyze-rec-errs filter-posts-by-err \
		   lin-train-xent-hardlab-perutt lin-nnet-forward lin-init lin-merge lin-init-ae lin-train-mse-ae \
		   linrbm-init linrbm-train-cd1-frmshuff linrbm-extract-linbl gen-2d-feats gmm-pdf-info \
		   nnet-train-mse-tgtmat-frmshuff-denoise write-ubm-to-matlab ubm-info grbm-vts-forward \
		   analyze-gmm-nnet hmmbl-create hmmbl-vts-forward normalize-feats nnet-train-xent-mse-frmshuff \
		   nnet-xent-mse-split nnet2-train-xent-mse-frmshuff ubm-avg-likes nnet-hidmask-train-frmshuff \
		   nnet-cleanh-train-frmshuff codebl-create codebl-train-xent-hardlab-frmshuff codevec-init \
		   codevec-train-xent-hardlab-frmshuff codebl-forward ideal-hidmask-forward lin-train-perutt-single-iter \
		   scale-nnet nnet-hidmask-mse-tgtmat-frmshuff nnet-hidmask-forward ideal-hidmask-stats nnet-train-stereo

OBJFILES =

all:  $(BINFILES)


TESTFILES =


$(BINFILES): ../nnet/kaldi-nnet.a ../cudamatrix/cuda-matrix.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a ../vts/kaldi-vts.a 



# Rule below would expand to, e.g.:
# ../base/kaldi-base.a:
# 	make -c ../base kaldi-base.a
# -c option to make is same as changing directory.
%.a:
	$(MAKE) -C ${@D} ${@F}

clean:
	rm *.o $(TESTFILES) $(BINFILES)

test: $(TESTFILES)
	for x in $(TESTFILES); do ./$$x >/dev/null 2>&1 || { echo "***test $$x failed***"; exit 1; }; done  
	echo Tests succeeded

.valgrind: $(TESTFILES)


depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > .depend.mk  

# removing automatic making of "depend" as it's quite slow.
#.depend.mk: depend

-include .depend.mk

