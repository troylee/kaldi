#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

#
# Train utterance dependent LIN
#

# Begin configuration.
nj=4
cmd=run.pl
config=
# NETWORK INITIALIZATION
nnet_lin=     #initialized nnet, with an LIN layer
feature_transform=        # provide additional feature transform

srcdir=

# LIN PARAMETERS
lin_init_dir=

# FEATURE PROCESSING
norm_vars=true    # use variance normalization
splice=5          # temporal splicing

# LABELS
labels=           # these labels to train
# or can directly provide alignment dir
alidir=
# or use recognized transcription
tra=
# options for alignment of tra
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=200

# TRAINING SCHEDULER
learn_rate=0.001
l1_penalty=0.0
l2_penalty=0.0
momentum=0.0
average_grad=true

train_tool="lin-train-perutt-single-iter"

# End configuration

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data> <lang-dir> <dir>"
   echo " e.g.: $0 data/train data/lang exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

data=$1
lang=$2
dir=$3

mkdir -p $dir/log

[ -z $srcdir ] && srcdir=`dirname $dir`; # Default model directory
[ -z $nnet_lin ] && nnet_lin=$dir/lin.init # default LIN nnet

# 
[ ! -z $lin_init_dir ] && [ "$nj" -ne "`cat $lin_init_dir/num_jobs`" ] && echo "Mismatch in #jobs with si-dir" && exit 1;
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z $labels ] && [ -z $alidir ] && [ -z $tra ] ; then
  echo "No label specified" && exit 1;
fi

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

for f in $sdata/1/feats.scp $nnet_lin ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Estimating per-utterance LIN Transforms"
printf "\t weight       : $lin_weight_tgt \n"
printf "\t bias         : $lin_bias_tgt \n"

# PREPARE FEATURE EXTRACTION PIPELINE
# Create the feature stream:
feats="ark,s,cs:add-deltas --delta-order=2 --delta-window=3 scp:$sdata/JOB/feats.scp ark:- |"
# Optionally add cmvn
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $sdata/1/cmvn_0_d_a.utt.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn_0_d_a.utt.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars scp:$sdata/JOB/cmvn_0_d_a.utt.scp ark:- ark:- |"
fi
# Optionally add splice
if [ -f $srcdir/splice ]; then
  splice=$(cat $srcdir/splice 2>/dev/null)
  feats="$feats splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
fi


###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"
if [ ! -z $labels ]; then
  printf "\t Train-set : $data $labels \n"
  echo "Using targets '$labels' (by force)"
  labels="$labels"
else 
  if [ ! -z $alidir ]; then
    printf "\t Train-set : $data $alidir \n"
    echo "Using PDF targets from dirs '$alidir'"
    for f in $alidir/final.mdl $alidir/ali.1.gz ; do
      [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
    done
    # define pdf-alignment rspecifiers
    labels="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
  else 
    printf "\t Train-set : $data $tra \n"
    echo "Using PDF targets from recognized hypotheses '$tra'"
    for f in $srcdir/final.mdl $srcdir/tree $srcdir/final.nnet $srcdir/ali_train_pdf.counts ; do
      [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
    done
    # features for alignment
    feats_ali="$feats nnet-forward ${feature_transform:+ --feature-transform=$feature_transform} --no-softmax=false --apply-log=true --class-frame-counts=$srcdir/ali_train_pdf.counts $srcdir/final.nnet ark:- ark:- |"
    # generate the alignments
    compile-train-graphs $srcdir/tree $srcdir/final.mdl $lang/L.fst "$tra" ark:- |
    align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $srcdir/final.mdl ark:- \
      "$feats_ali" "ark,t:|gzip -c >$dir/ali.gz" || exit 1;
    # prepare labels
    labels="ark:ali-to-pdf $srcdir/final.mdl \"ark:gunzip -c $dir/ali.gz |\" ark:- |"
  fi
fi


###### PREPARE ALIGNMENTS ######
echo "LIN TRAINING..."
$cmd JOB=1:$nj $dir/log/train_lin.JOB.log \
  $train_tool --average-grad=${average_grad} --learn-rate=${learn_rate} --momentum=${momentum} \
  --l1-penalty=${l1_penalty} --l2-penalty=${l2_penalty} \
  ${lin_weight_init:+ --weight-init="ark:$lin_init_dir/lin_weight.JOB.ark"} \
  ${lin_bias_init:+ --bias-init="ark:$lin_init_dir/lin_bias.JOB.ark"} \
  ${feature_transform:+ --feature-transform=$feature_transform} \
  ${nnet_lin} "${feats}" "${labels}" ark,scp:$dir/lin_weight.JOB.ark,$dir/lin_weight.JOB.scp \
  ark,scp:$dir/lin_bias.JOB.ark,$dir/lin_bias.JOB.scp || exit 1;


sleep 3;
echo "$0: successfully done "

