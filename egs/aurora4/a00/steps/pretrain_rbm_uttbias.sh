#!/bin/bash
# Copyright 2014 Bo Li
# Apache 2.0

#
# RBM training with utt-bias
#

# Begin configuration.
feature_transform=
iters=1
lrate=0.001
momentum=0.5
l2_penalty=0.0002

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/pretrain_rbm.sh <rbm-in> <feature-pipeline> <rbm-out>"
   echo " e.g.: steps/pretrain_rbm.sh rbm.init \"scp:features.scp\" rbm.trained"
   exit 1;
fi

rbm_in=$1
features=$2
rbm_out=$3

if [ ! -f $rbm_in ]; then
  echo "Error: initial rbm '$rbm_in' does not exist"
  exit 1;
fi

######## CONFIGURATION
TRAIN_TOOL="rbm-uttbias-train"

dir=$(dirname $rbm_out)
logdir=$dir/../log
mkdir -p $dir $logdir

base=$(basename $rbm_out)

#PRE-TRAIN
# the 1st iteration is different
$TRAIN_TOOL --learn-rate=$lrate --momentum=$momentum --l2-penalty=$l2penalty \
  ${feature_transform:+ --feature-transform=$feature_transform} \
  "$features" ark:$dir/visbias.1.ark ark:$dir/hidbias.1.ark $rbm_in $rbm_out.iter1 2>$logdir/$base.iter1.log || exit 1

for i in $(seq 2 $iters); do
  $TRAIN_TOOL --learn-rate=$lrate --momentum=$momentum --l2-penalty=$l2penalty \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    --init-visbias=ark:$dir/visbias.$((i-1)).ark \
    --init-hidbias=ark:$dir/hidbias.$((i-1)).ark \
    "$features" ark:$dir/visbias.${i}.ark ark:$dir/hidbias.${i}.ark $rbm_in $rbm_out.iter$i 2>$logdir/$base.iter$i.log || exit 1
  rm $rbm_in $dir/visbias.$((i-1)).ark $dir/hidbias.$((i-1)).ark
  rbm_in=$rbm_out.iter$i
done

#make full path
[[ ${rbm_out:0:1} != "/" && ${rbm_out:0:1} != "~" ]] && rbm_out=$PWD/$rbm_out
[[ ${dir:0:1} != "/" && ${dir:0:1} != "~" ]] && dir=$PWD/$dir

#link the final rbm
ln -s $rbm_out.iter$i $rbm_out
ln -s $dir/visbias.${i}.ark $dir/visbias.ark
ln -s $dir/hidbias.${i}.ark $dir/hidbias.ark

echo "$0 finished ok"


