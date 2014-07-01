#!/bin/bash

# Copyright 2014  Bo Li
# Apache 2.0

# Begin configuration section.  
transform_dir=
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
stage=0
nj=4
cmd=run.pl
max_active=15000
beam=30.0
latbeam=9.0
acwt=0.0625 # note: only really affects pruning (scoring is on lattices).

num_fbank=26
num_cepstral=13
ceplifter=22

variance_lr=1.0
max_noise_mean_magnitude=1000.0
noise_iterations=2
em_iterations=8

scoring_opts=
# note: there are no more min-lmwt and max-lmwt options, instead use
# e.g. --scoring-opts "--min-lmwt 1 --max-lmwt 20"
skip_scoring=false
srcdir=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --srcdir <dir>                                   # non default model file directory"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --noise-iterations <n>                           # number of noise estimations, default 1."
   echo "  --em-iterations <n>                              # number of EM iterations, default 8"
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

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; 
  else model=$srcdir/$iter.mdl; fi
fi

for f in $sdata/1/feats.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode.sh: no such file $f" && exit 1;
done 

# no cmvn or transformations for vts
feats="ark,s,cs:add-deltas --delta-order=2 --delta-window=3 scp:$sdata/JOB/feats.scp ark:- |"

# decoding for noise 
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    vts-model-decode --variance-lrate=$variance_lr --max-noise-mean-magnitude=1000.0 \
    --noise-iterations=$noise_iterations --em-iterations=$em_iterations \
    --num-fbank=$num_fbank --num-cepstral=$num_cepstral --ceplifter=$ceplifter \
    --max-active=$max_active --beam=$beam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst "$feats" ark,t:$dir/rec.JOB.tra ark,t:$dir/noise.JOB.ark || exit 1;
fi

# generate lattice
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/latgen.JOB.log \
    vts-noise-latgen --max-active=$max_active --beam=$beam --lattice-beam=$latbeam \
    --num-fbank=$num_fbank --num-cepstral=$num_cepstral --ceplifter=$ceplifter \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst "$feats" ark:$dir/noise.JOB.ark "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir
fi

exit 0;

