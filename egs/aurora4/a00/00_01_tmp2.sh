#!/bin/bash
# Author: Bo Li (li-bo@outlook.com)
#
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
#

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh ## update system path

log_start(){
  echo "#####################################################################"
  echo "Spawning *** $1 *** on" `date` `hostname`
  echo ---------------------------------------------------------------------
}

log_end(){
  echo ---------------------------------------------------------------------
  echo "Done *** $1 *** on" `date` `hostname` 
  echo "#####################################################################"
}

# experiment with dropout fine-tuning
train_dnn1d_dptmp(){
  log_start "dnn1d_dptmp [train]"
  dir=exp_multi/dnn1d_dptmp
  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  ori_mlp=exp_multi/dnn1b/nnet.init
  mkdir -p $dir/log
  nnet-add-dropout --add-to-input=false --binary=false \
    --hidden-drop-ratio=0.3 $ori_mlp $dir/nnet.init 1 3 5 7 9 11 || exit 1;
  steps/train_nnet_dropout.sh --norm-vars true --splice 5 --mlp-init $dir/nnet.init \
    --alidir $ali --alidir-cv $ali_dev --debug true \
    --bunchsize 256 --l2-upperbound 1.0 --average-grad true \
    --high-learn-rate 0.005 --low-learn-rate 0.001 \
    --momentum-init 0.1 --momentum-inc 0.05 --num-iters-momentum-adjust 8 \
    --momentum-low 0.9 --momentum-high 0.9 --iter-shuffle true \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "dnn1d_dptmp [train]"
}
train_dnn1d_dptmp

