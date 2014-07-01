#!/bin/bash

# single pass retraining to obtain models using unnormalized features

# Begin configuration
config=
cmd=run.pl
cmvn_opts=
# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/aurora4_singlepass_retrain.sh <data-dir> <alignment-dir> <exp-dir>"
   echo "e.g.: steps/aurora4_singlepass_retrain.sh data/train_si84_half exp/tri1a_ali exp/tri1b"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   exit 1;
fi

data=$1
alidir=$2
dir=$3

for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp ; do
  [ ! -f $f ] && echo "singlepass_retrain.sh: no such file $f" && exit 1;
done


# tri1b, single pass retrain from tri1a, using un-normalized MFCC features
nj=`cat $alidir/num_jobs` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs

sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

if [ -z "$cmvn_opts" ] && [ -f $alidir/cmvn_opts ] && [ $(cat $alidir/cmvn_opts 2>/dev/null | wc -l ) -eq 1 ]; then
  echo "using $alidir/cmvn_opts"
  cmvn_opts=`cat $alidir/cmvn_opts`
fi

# features
oldfeats="ark,s,cs:add-deltas --delta-order=2 --delta-window=3 scp:$sdata/JOB/feats.scp ark:- | apply-cmvn $cmvn_opts scp:$sdata/JOB/cmvn_0_d_a.utt.scp ark:- ark:- |"
newfeats="ark,s,cs:add-deltas --delta-order=2 --delta-window=3 scp:$sdata/JOB/feats.scp ark:- |"


echo "$0: accumulate two feats stats"
$cmd JOB=1:$nj $dir/log/acc.JOB.log \
	gmm-acc-stats-twofeats $alidir/final.mdl "$oldfeats" "$newfeats" "ark:ali-to-post \"ark:gunzip -c $alidir/ali.JOB.gz |\" ark:- |" $dir/JOB.acc || exit 1;

# Update model.
echo "$0: update model"
$cmd $dir/log/update.log \
	gmm-est --write-occs=${dir}/final.occs --remove-low-count-gaussians=false $alidir/final.mdl \
		"gmm-sum-accs - $dir/*.acc |" $dir/final.mdl || exit 1;

cp $alidir/tree ${dir}/tree

