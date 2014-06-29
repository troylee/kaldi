#!/bin/bash 

# Compute per utterance CMVN stats for 0_D_A features

echo "$0 $@"  # Print the command line for logging

fake=false
two_channel=false

if [ $1 == "--fake" ]; then
  fake=true
  shift
fi

if [ $# != 3 ]; then
   echo "usage: compute_cmvn_utt_stats.sh [options] <data-dir> <log-dir> <path-to-cmvn-dir>";
   echo "Options:"
   echo " --fake          gives you fake cmvn stats that do no normalization."
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

data=$1
logdir=$2
cmvndir=$3

# make $cmvndir an absolute pathname.
cmvndir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $cmvndir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $cmvndir || exit 1;
mkdir -p $logdir || exit 1;


required="$data/feats.scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_cmvn.sh: no such file $f"
    exit 1;
  fi
done

if $fake; then
  dim=`feat-to-dim scp:$data/feats.scp -`
  ! cat $data/feats.scp | awk -v dim=$dim '{print $1, "["; for (n=0; n < dim*3; n++) { printf("0 "); } print "1"; for (n=0; n < dim*3; n++) { printf("1 "); } print "0 ]";}' | \
    copy-matrix ark:- ark,scp:$cmvndir/cmvn_0_d_a.utt.$name.ark,$cmvndir/cmvn_0_d_a.utt.$name.scp && \
     echo "Error creating fake CMVN stats" && exit 1;
else
  ! add-deltas --delta-order=2 --delta-window=3 scp:$data/feats.scp ark:- | compute-cmvn-stats ark:- ark,scp:$cmvndir/cmvn_0_d_a.utt.$name.ark,$cmvndir/cmvn_0_d_a.utt.$name.scp \
    2> $logdir/cmvn_0_d_a.utt.$name.log && echo "Error computing CMVN stats" && exit 1;
fi

cp $cmvndir/cmvn_0_d_a.utt.$name.scp $data/cmvn_0_d_a.utt.scp || exit 1;

nc=`cat $data/cmvn_0_d_a.utt.scp | wc -l` 
nu=`cat $data/feats.scp | wc -l` 
if [ $nc -ne $nu ]; then
  echo "$0: warning: it seems not all of the utterances got cmvn stats ($nc != $nu);"
  [ $nc -eq 0 ] && exit 1;
fi

echo "Succeeded creating CMVN stats for $name"

