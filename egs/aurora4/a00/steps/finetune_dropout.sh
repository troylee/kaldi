#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

# Train neural network, halving of the learning rate is done only when the improvement 
# is too low 

# Begin configuration.

# training options
learn_rate=0.008
momentum_init=0
momentum_inc=0
l1_penalty=0
l2_penalty=0
average_grad=
# data processing
bunchsize=256
cachesize=32768
feature_transform=
# learn rate scheduling
num_iters=1
# misc.
verbose=1
# tool
train_tool="nnet-train-xent-hardlab-frmshuff" 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 <feats-tr> <feats-cv> <labels-tr> <labels-cv> <nnet-in> <nnet-out>"
   echo " e.g.: $0 scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark 0.nnet 1.nnet"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi


feats_tr=$1
feats_cv=$2
labels_tr=$3
labels_cv=$4
nnet_in=$5
nnet_out=$6

dir=$(dirname $nnet_out)
logdir=$dir/../log
[ ! -d $dir ] && mkdir $dir
[ ! -d $logdir ] && mkdir $logdir

# Skip training
[ -e $nnet_out ] && echo "'$nnet_out' exists, skipping training" && exit 0

##############################
#start training

base=$(basename $nnet_out)
momentum=$momentum_init
#Dropout tuning
for iter in $(seq 1 $num_iters); do
  echo -n "ITERATION $iter: "
  # training
  log=$logdir/${base}.iter${iter}.tr.log; hostname>$log
  $train_tool \
    --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
    --bunchsize=$bunchsize --cachesize=$cachesize --randomize=true --verbose=$verbose \
    --binary=true \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${average_grad:+ "--average-grad=$average_grad"} \
    "$feats_tr" "$labels_tr" $nnet_in $nnet_out.iter${iter} \
    2> $log || exit 1; 

  tr_acc=$(cat $log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "TRAIN AVG.FRMACC $(printf "%.4f" $tr_acc), (lrate$(printf "%.6g" $learn_rate)), "
  
  # cross-validation
  log=$logdir/${base}.iter${iter}.cv.log; hostname>$log
  nnet-rm-dropout --binary=true $nnet_out.iter${iter} $nnet_out.iter${iter}.fwd 2>$log || exit 1;
  $train_tool --cross-validate=true \
    --bunchsize=$bunchsize --cachesize=$cachesize --verbose=$verbose \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    "$feats_cv" "$labels_cv" $nnet_out.iter${iter}.fwd \
    2>>$log || exit 1;
  
  acc_new=$(cat $log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "CROSSVAL AVG.FRMACC $(printf "%.4f" $acc_new), "

  [ ${iter} -gt 1 ] && rm $nnet_in
  rm $nnet_out.iter${iter}.fwd
  nnet_in=$nnet_out.iter${iter}
  # update momentum
  momentum=`perl -e "print ($momentum + $momentum_inc);"`
done

#make full path
[[ ${nnet_out:0:1} != "/" && ${nnet_out:0:1} != "~" ]] && nnet_out=$PWD/$nnet_out

#link the final nnet
ln -s $nnet_out.iter${iter} $nnet_out

echo "$0 finished ok"

