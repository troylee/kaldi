#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

# Begin configuration section. 
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
model=              # non-default location of transition model (optional)
class_frame_counts= # non-default location of PDF counts (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl

# RBM UTT opts
rbm_mdl=
hidbias=

acwt=`perl -e "print (1.0/17.0);"` # note: only really affects pruning (scoring is on lattices).
beam=15.0
latbeam=9.0
min_active=200
max_active=15000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

skip_scoring=false
scoring_opts="--min-lmwt 9 --max-lmwt 20"

num_threads=1 # if >1, will use latgen-faster-parallel
parallel_opts="" # use 2 CPUs (1 DNN-forward, 1 decoder)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the DNN and transition model is."
   echo "e.g.: $0 exp/dnn1/graph_tgpr data/test exp/dnn1/decode_tgpr"
   echo ""
   echo "This script works on plain or modified features (CMN,delta+delta-delta),"
   echo "which are then sent through feature-transform. It works out what type"
   echo "of features you used from content of srcdir."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --srcdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   echo ""
   echo "  --acwt <float>                                   # select acoustic scale for decoding"
   echo "  --scoring-opts <opts>                            # options forwarded to local/score.sh"
   echo "  --num-threads <N>                                # N>1: run multi-threaded decoder"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
[ -z $srcdir ] && srcdir=`dirname $dir`; # Default model directory one level up from decoding directory.
sdata=$data/split$nj;

[ -z $rbm_mdl ] && echo "No RBM(uttbias) model specified!" && exit 1;

mkdir -p $dir/log

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Select default locations to model files (if not already set externally)
if [ -z "$nnet" ]; then nnet=$srcdir/final.nnet; fi
if [ -z "$model" ]; then model=$srcdir/final.mdl; fi
if [ -z "$class_frame_counts" ]; then class_frame_counts=$srcdir/ali_train_pdf.counts; fi

# Check that files exist
for f in $sdata/1/feats.scp $nnet $model $class_frame_counts $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 

# PREPARE FEATURE EXTRACTION PIPELINE
# Create the feature with deltas
feats="ark:add-deltas --delta-order=2 --delta-window=3 scp:$sdata/JOB/feats.scp ark:- |"
# add CMVN
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $sdata/1/cmvn_0_d_a.utt.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn_0_d_a.utt.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars scp:$sdata/JOB/cmvn_0_d_a.utt.scp ark:- ark:- |"
fi
# splicing 
if [ -f $srcdir/splice ]; then
  splice=$(cat $srcdir/splice 2>/dev/null)
  feats="$feats splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
fi

# Run the decoding in the queue
if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    rbmdnn-forward ${feature_transform:+ --feature-transform="$feature_transform"} \
    ${hidbias:+ --hidbias="$hidbias"} --rbm-binarize=false --rbm-apply-log=false \
    --no-softmax=false --apply-log=true --class-frame-counts=$class_frame_counts $rbm_mdl $nnet "$feats" ark:- \| \
    latgen-faster-mapped$thread_string --max-active=$max_active --max-mem=$max_mem --beam=$beam \
    --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

# Run the scoring
if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
fi

exit 0;

