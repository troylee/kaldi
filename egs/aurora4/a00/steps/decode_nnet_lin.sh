#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

# Begin configuration section. 
lin_nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
model=              # non-default location of transition model (optional)
class_frame_counts= # non-default location of PDF counts (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)

# directory contains LIN weights and biases
lindir=

tra=    # decoding hypotheses for estimating LINs
tra_align_dir=    # models used for converting the $tra to pdf alignment
tra_lang=data/lang
tra_scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
tra_beam=10
tra_retry_beam=200
learn_rate=0.001
l1_penalty=0.0
l2_penalty=0.0
momentum=0.0
average_grad=true

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl

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

mkdir -p $dir/log

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Select default locations to model files (if not already set externally)
if [ -z "$lin_nnet" ]; then lin_nnet=$srcdir/final_lin.nnet; fi
if [ -z "$model" ]; then model=$srcdir/final.mdl; fi
if [ -z "$class_frame_counts" ]; then class_frame_counts=$srcdir/ali_train_pdf.counts; fi

# Check that files exist
for f in $sdata/1/feats.scp $lin_nnet $model $class_frame_counts $graphdir/HCLG.fst; do
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

# PREPARE LIN PARAMETERS
if [ -z $lindir ]; then
  [ -z $tra ] && echo "decoding hypotheses are required to estimate LINs" && exit 1;
  [ -z $tra_align_dir ] && echo "directory for models used for alignment not specified" && exit 1;
  # prepare alignment
  for f in $tra_align_dir/final.mdl $tra_align_dir/tree $tra_align_dir/final.nnet $tra_align_dir/ali_train_pdf.counts ; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
  # features for alignment
  feats_ali="ark:add-deltas --delta-order=2 --delta-window=3 scp:$data/feats.scp ark:- |"
  # add CMVN
  if [ -f $tra_align_dir/norm_vars ]; then
    norm_vars=$(cat $tra_align_dir/norm_vars 2>/dev/null)
    [ ! -f $data/cmvn_0_d_a.utt.scp ] && echo "$0: cannot find cmvn stats $data/cmvn_0_d_a.utt.scp" && exit 1
    feats_ali="$feats_ali apply-cmvn --norm-vars=$norm_vars scp:$data/cmvn_0_d_a.utt.scp ark:- ark:- |"
  fi
  # splicing 
  if [ -f $tra_align_dir/splice ]; then
    splice=$(cat $tra_align_dir/splice 2>/dev/null)
    feats_ali="$feats_ali splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
  fi
  feats_ali="$feats_ali nnet-forward ${feature_transform:+ --feature-transform=$feature_transform} --no-softmax=false --apply-log=true --class-frame-counts=$tra_align_dir/ali_train_pdf.counts $tra_align_dir/final.nnet ark:- ark:- |"
  # generate the alignments
  compile-train-graphs $tra_align_dir/tree $tra_align_dir/final.mdl $tra_lang/L.fst ark:$tra ark:- |
  align-compiled-mapped $tra_scale_opts --beam=$tra_beam --retry-beam=$tra_retry_beam $tra_align_dir/final.mdl ark:- \
    "$feats_ali" "ark,t:|gzip -c >$dir/ali.gz" || exit 1;
  # prepare labels
  labels="ark:ali-to-pdf $tra_align_dir/final.mdl \"ark:gunzip -c $dir/ali.gz |\" ark:- |"
  # estimate LINs first
  $cmd JOB=1:$nj $dir/log/estimate_lin.JOB.log \
    lin-train-perutt-single-iter --average-grad=${average_grad} --learn-rate=${learn_rate} --momentum=${momentum} \
    --l1-penalty=${l1_penalty} --l2-penalty=${l2_penalty} \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${lin_nnet} "${feats}" "${labels}" ark,scp:$dir/lin_weight.JOB.ark,$dir/lin_weight.JOB.scp \
    ark,scp:$dir/lin_bias.JOB.ark,$dir/lin_bias.JOB.scp || exit 1;
  # curdir is lindir
  lindir=$dir
fi
# ensure the number of jobs are the same
[ "`cat $lindir/num_jobs`" -ne $nj ] && echo "Job number mismatch!" && exit 1;

# Run the decoding in the queue
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    lin-nnet-forward ${feature_transform:+ --feature-transform="$feature_transform"} \
    --no-softmax=false --apply-log=true --class-frame-counts=$class_frame_counts \
    $lin_nnet ark:$lindir/lin_weight.JOB.ark ark:$lindir/lin_bias.JOB.ark \
    "$feats" ark:- \| \
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


