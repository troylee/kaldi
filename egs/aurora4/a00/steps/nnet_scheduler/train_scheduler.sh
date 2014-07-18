#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

# Train neural network, halving of the learning rate is done only when the improvement 
# is too low 

# Begin configuration.

# training options
learn_rate=0.008
momentum=0
l1_penalty=0
l2_penalty=0
average_grad=
# data processing
bunchsize=256
cachesize=32768
feature_transform=
# learn rate scheduling
max_iters=99
min_iters=
start_halving_impr=0.01
halving_factor=0.5
end_learn_rate=0.000001

learn_factors=

accept_first_update=false

# misc.
verbose=1
# tool
train_tool="nnet-train-xent-hardlab-frmshuff"
frame_weights=
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
dir=$6

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
#start training

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
# optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

# cross-validation on original network
log=$dir/log/iter00.initial.log; hostname>$log
$train_tool --cross-validate=true \
 --bunchsize=$bunchsize --cachesize=$cachesize --verbose=$verbose \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 "$feats_cv" "$labels_cv" $mlp_best \
 2>> $log || exit 1;

acc=$(cat $dir/log/iter00.initial.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
echo "CROSSVAL PRERUN AVG.FRMACC $(printf "%.4f" $acc)"
$accept_first_update && acc=0.0

# training
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  # skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  # training
  log=$dir/log/iter${iter}.tr.log; hostname>$log
  $train_tool \
   --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
   --bunchsize=$bunchsize --cachesize=$cachesize --randomize=true --verbose=$verbose \
   --binary=true \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   ${learn_factors:+ --learn-factors=$learn_factors} \
   ${average_grad:+ "--average-grad=$average_grad"} \
   "$feats_tr" "$labels_tr" $mlp_best $mlp_next \
   2>> $log || exit 1; 

  tr_acc=$(cat $dir/log/iter${iter}.tr.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "TRAIN AVG.FRMACC $(printf "%.4f" $tr_acc), (lrate$(printf "%.6g" $learn_rate)), "
  
  # cross-validation
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --cross-validate=true \
   --bunchsize=$bunchsize --cachesize=$cachesize --verbose=$verbose \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   "$feats_cv" "$labels_cv" $mlp_next \
   2>>$log || exit 1;
  
  acc_new=$(cat $dir/log/iter${iter}.cv.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "CROSSVAL AVG.FRMACC $(printf "%.4f" $acc_new), "

  # accept or reject new parameters (based on objective function)
  acc_prev=$acc
  if [ "1" == "$(awk "BEGIN{print($acc_new>$acc);}")" ]; then
    acc=$acc_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_acc)_cv$(printf "%.4f" $acc_new)
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_acc)_cv$(printf "%.4f" $acc_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter

  # stopping criterion
   if [[ "1" == "$(awk "BEGIN{print($learn_rate < $end_learn_rate)}")" ]]; then
    if [[ "$min_iters" != "" ]]; then
      if [ $min_iters -gt $iter ]; then
        echo we were supposed to finish, but we continue, min_iters : $min_iters
        continue
      fi
    fi
    echo finished, too small rel. improvement $(awk "BEGIN{print($acc-$acc_prev)}")
    break
  fi

  # start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print($acc < $acc_prev+$start_halving_impr)}")" ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi
done

# select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi

