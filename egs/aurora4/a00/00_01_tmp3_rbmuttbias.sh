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

######################################
# experiments with RBM(uttbias)

pretrain_rub1a(){
  #RBM pretrain
  log_start "rub1a [pretrain]"
  dir=exp_multi/rub1a_pretrain
  rbm_init=exp_multi/dnn1a_pretrain/hid01_rbm/nnet/hid01_rbm_mmt0.9.iter200
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/train_multi $dir
  log_end "rub1a [pretrain]"
}
pretrain_rub1a

