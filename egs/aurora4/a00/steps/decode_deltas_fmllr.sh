#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

# Decoding script that does fMLLR. 

# There are 3 models involved potentially in this script,
# and for a standard, speaker-independent system they will all be the same.
# The "alignment model" is for the 1st-pass decoding and to get the 
# Gaussian-level alignments for the "adaptation model" the first time we
# do fMLLR.  The "adaptation model" is used to estimate fMLLR transforms
# and to generate state-level lattices.  The lattices are then rescored
# with the "final model".
#
# The following table explains where we get these 3 models from.
# Note: $srcdir is one level up from the decoding directory.
#
#   Model              Default source:                 
#
#  "alignment model"   $srcdir/final.alimdl              --alignment-model <model>
#                     (or $srcdir/final.mdl if alimdl absent)
#  "adaptation model"  $srcdir/final.mdl                 --adapt-model <model>
#  "final model"       $srcdir/final.mdl                 --final-model <model>


# Begin configuration section
first_beam=10.0 # Beam used in initial, speaker-indep. pass
first_max_active=2000 # max-active used in first two passes.
first_latbeam=4.0 # lattice pruning beam for si decode and first-pass fMLLR decode.
                # the different spelling from lattice_beam is unfortunate; these scripts
                # have a history.
alignment_model=
adapt_model=
final_model=
cleanup=true
stage=0
acwt=0.083333 # Acoustic weight used in getting fMLLR transforms, and also in 
              # lattice generation.
max_active=7000
beam=13.0
lattice_beam=6.0
nj=4
silence_weight=0.01
distribute=true  # option to weight-silence-post.
cmd=run.pl

# SI decoding directory
si_dir=
# model directory 
srcdir=

fmllr_update_type=full
skip_scoring=false
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
parallel_opts=  # If you supply num-threads, you should supply this too.
scoring_opts=

per_speaker=true # otherwise per-utterance
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode_fmllr.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_fmllr.sh exp/tri2b/graph_tgpr data/test_dev93 exp/tri2b/decode_dev93_tgpr"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                   # config containing options"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --adapt-model <adapt-mdl>                # Model to compute transforms with"
   echo "  --alignment-model <ali-mdl>              # Model to get Gaussian-level alignments for"
   echo "                                           # 1st pass of transform computation."
   echo "  --final-model <finald-mdl>               # Model to finally decode with"
   echo "  --si-dir <speaker-indep-decoding-dir>    # use this to skip 1st pass of decoding"
   echo "                                           # Caution-- must be with same tree"
   echo "  --acwt <acoustic-weight>                 # default 0.08333 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 1."
   echo "  --parallel-opts <opts>                   # e.g. '-pe smp 4' if you supply --num-threads 4"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi


graphdir=$1
data=$2
dir=`echo $3 | sed 's:/$::g'` # remove any trailing slash.

[ -z $srcdir ] && srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
norm_vars=`cat $srcdir/norm_vars 2>/dev/null` || norm_vars=false # cmn/cmvn option, default false.

silphonelist=`cat $graphdir/phones/silence.csl` || exit 1;

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $srcdir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Work out name of alignment model. ##
if [ -z "$alignment_model" ]; then
  if [ -f "$srcdir/final.alimdl" ]; then alignment_model=$srcdir/final.alimdl;
  else alignment_model=$srcdir/final.mdl; fi
fi
[ ! -f "$alignment_model" ] && echo "$0: no alignment model $alignment_model " && exit 1;
##

## Do the speaker-independent decoding, if --si-dir option not present. ##
if [ -z "$si_dir" ]; then # we need to do the speaker-independent decoding pass.
  si_dir=${dir}.si # Name it as our decoding dir, but with suffix ".si".
  if [ $stage -le 0 ]; then
    steps/decode_deltas.sh --acwt $acwt --nj $nj --cmd "$cmd" --beam $first_beam --model $alignment_model \
      --max-active $first_max_active --parallel-opts "${parallel_opts}" --num-threads $num_threads \
      --skip-scoring true $graphdir $data $si_dir || exit 1;
  fi
fi
##

## Some checks, and setting of defaults for variables.
[ "$nj" -ne "`cat $si_dir/num_jobs`" ] && echo "Mismatch in #jobs with si-dir" && exit 1;
[ ! -f "$si_dir/lat.1.gz" ] && echo "No such file $si_dir/lat.1.gz" && exit 1;
[ -z "$adapt_model" ] && adapt_model=$srcdir/final.mdl
[ -z "$final_model" ] && final_model=$srcdir/final.mdl
for f in $adapt_model $final_model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
##

utt2spk_opt=""
spk2utt_opt=""
$per_speaker && ( utt2spk_opt="--utt2spk=ark:$sdata/JOB/utt2spk"; spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"; )

## Set up the unadapted features "$sifeats"
sifeats="ark,s,cs:add-deltas --delta-order=2 --delta-window=3 scp:$sdata/JOB/feats.scp ark:- |"
# add cmvn if exists
if [ -f $srcdir/cmvn_opts ]; then
  cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
  sifeats="$sifeats apply-cmvn $cmvn_opts scp:$sdata/JOB/cmvn_0_d_a.utt.scp ark:- ark:- |"
fi

if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne $nj ] && \
     echo "Mismatch in number of jobs with $transform_dir";
  sifeats="$sifeats transform-feats $utt2spk_opt ark:$transform_dir/trans.JOB ark:- ark:- |"
fi
##

## Now get the first-pass fMLLR transforms.
if [ $stage -le 1 ]; then
  echo "$0: getting first-pass fMLLR transforms."
  $cmd JOB=1:$nj $dir/log/fmllr_pass1.JOB.log \
    gunzip -c $si_dir/lat.JOB.gz \| \
    lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
    weight-silence-post --distribute=$distribute $silence_weight $silphonelist $alignment_model ark:- ark:- \| \
    gmm-post-to-gpost $alignment_model "$sifeats" ark:- ark:- \| \
    gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
    $spk2utt_opt $adapt_model "$sifeats" ark,s,cs:- \
    ark:$dir/trans1.JOB || exit 1;
fi
##

pass1feats="$sifeats transform-feats $utt2spk_opt ark:$dir/trans1.JOB ark:- ark:- |"

## Do the first adapted lattice generation pass. 
if [ $stage -le 2 ]; then
  echo "$0: doing first adapted lattice generation phase"
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode1.JOB.log\
    gmm-latgen-faster$thread_string --max-active=$first_max_active --beam=$first_beam --lattice-beam=$first_latbeam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $adapt_model $graphdir/HCLG.fst "$pass1feats" "ark:|gzip -c > $dir/lat1.JOB.gz" \
    || exit 1;
fi


## Do a second pass of estimating the transform.  Compose the transforms to get
## $dir/trans2.*.
if [ $stage -le 3 ]; then
  echo "$0: estimating fMLLR transforms a second time."
  $cmd JOB=1:$nj $dir/log/fmllr_pass2.JOB.log \
    lattice-to-post --acoustic-scale=$acwt "ark:gunzip -c $dir/lat1.JOB.gz|" ark:- \| \
    weight-silence-post --distribute=$distribute $silence_weight $silphonelist $adapt_model ark:- ark:- \| \
    gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
    $spk2utt_opt $adapt_model "$pass1feats" \
    ark,s,cs:- ark:$dir/trans1b.JOB '&&' \
    compose-transforms --b-is-affine=true ark:$dir/trans1b.JOB ark:$dir/trans1.JOB \
    ark:$dir/trans2.JOB  || exit 1;
  if $cleanup; then
    rm $dir/trans1b.* $dir/trans1.* $dir/lat1.*.gz
  fi
fi
##

pass2feats="$sifeats transform-feats $utt2spk_opt ark:$dir/trans2.JOB ark:- ark:- |"

# Generate a 3rd set of lattices, with the "adaptation model"; we'll use these
# to adapt a 3rd time, and we'll rescore them.  Since we should be close to the final
# fMLLR, we don't bother dumping un-determinized lattices to disk.

## Do the final lattice generation pass (but we'll rescore these lattices
## after another stage of adaptation.)
if [ $stage -le 4 ]; then
  echo "$0: doing final lattice generation phase"
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode2.JOB.log\
    gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $adapt_model $graphdir/HCLG.fst "$pass2feats" "ark:|gzip -c > $dir/lat2.JOB.gz" \
    || exit 1;
fi


## Do a third pass of estimating the transform.  Compose the transforms to get
## $dir/trans.*.
if [ $stage -le 5 ]; then
  echo "$0: estimating fMLLR transforms a third time."
  $cmd JOB=1:$nj $dir/log/fmllr_pass3.JOB.log \
    lattice-to-post --acoustic-scale=$acwt "ark:gunzip -c $dir/lat2.JOB.gz|" ark:- \| \
    weight-silence-post --distribute=$distribute $silence_weight $silphonelist $adapt_model ark:- ark:- \| \
    gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
    $spk2utt_opt $adapt_model "$pass2feats" \
    ark,s,cs:- ark:$dir/trans2b.JOB '&&' \
    compose-transforms --b-is-affine=true ark:$dir/trans2b.JOB ark:$dir/trans2.JOB \
    ark:$dir/trans.JOB  || exit 1;
  if $cleanup; then
    rm $dir/trans2b.* $dir/trans2.*
  fi
fi
##

feats="$sifeats transform-feats $utt2spk_opt ark:$dir/trans.JOB ark:- ark:- |"

if [ $stage -le 6 ]; then
  echo "$0: doing a final pass of acoustic rescoring."
  $cmd JOB=1:$nj $dir/log/acoustic_rescore.JOB.log \
    gmm-rescore-lattice $final_model "ark:gunzip -c $dir/lat2.JOB.gz|" "$feats" \
      "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
  if $cleanup; then
    rm $dir/lat2.*.gz
  fi
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
fi

exit 0;

