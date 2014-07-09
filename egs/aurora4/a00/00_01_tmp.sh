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
train_dnn2d(){
  log_start "dnn2d [train]"
  dir=exp_multi/dnn2d
  ali=exp_multi/tri1a_ali/train_multi
  ali_dev=exp_multi/tri1a_ali/dev_multi
  ori_mlp=exp_multi/dnn2c/nnet.init
  mkdir -p $dir/log
#  cat $ori_mlp | sed "s:<sigmoid>:<relu>:g" > $dir/nnet.tmp || exit 1;
#  nnet-add-dropout --add-to-input=true --binary=false --input-drop-ratio=0.2 \
#    --hidden-drop-ratio=0.5 $dir/nnet.tmp $dir/nnet.init 1 3 5 7 9 11 || exit 1;
#  rm $dir/nnet.tmp
  steps/train_nnet_dropout.sh --norm-vars true --splice 5 --mlp-init $dir/nnet.init \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph_bg || exit 1;
  log_end "dnn2d [train]"
}
train_dnn2d

