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

##=================================================================
## Experiments with RBM for robustness
##=================================================================

##--------------------
## RBM with utterance dependent visible and hidden biases
pretrain_rub1a(){
  #RBM pretrain
  log_start "rub1a [pretrain]"
  dir=exp_multi/rub1a_pretrain
  rbm_init=exp_multi/dnn1a_pretrain/hid01_rbm/nnet/hid01_rbm_mmt0.9.iter200
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-visbias true --learn-hidbias true --debug true --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/train_multi $dir
  log_end "rub1a [pretrain]"
}
#pretrain_rub1a

pretrain_rub1a_dev(){
  #RBM pretrain
  log_start "rub1a [dev pretrain]"
  dir=exp_multi/rub1a_pretrain/dev_multi
  rbm_init=exp_multi/rub1a_pretrain/final.rbm
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-weight false --learn-visbias true --learn-hidbias true --debug false --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/dev_multi $dir || exit 1;
  log_end "rub1a [dev pretrain]"
}
#pretrain_rub1a_dev

pretain_rub1a_test(){
  #RBM pretrain
  log_start "rub1a [test pretrain]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    dir=exp_multi/rub1a_pretrain/test/$x
    rbm_init=exp_multi/rub1a_pretrain/final.rbm
    mkdir -p $dir/log
    steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-weight false --learn-visbias true --learn-hidbias true --debug false --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/$x $dir || exit 1;
  done
  log_end "rub1a [test pretrain]"
}
pretain_rub1a_test

##--------------------
## RBM with utterance dependent hidden biases only
pretrain_rub2a(){
  #RBM pretrain
  log_start "rub2a [pretrain]"
  dir=exp_multi/rub2a_pretrain
  rbm_init=exp_multi/dnn1a_pretrain/hid01_rbm/nnet/hid01_rbm_mmt0.9.iter200
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-visbias false --learn-hidbias true --debug true --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/train_multi $dir
  log_end "rub2a [pretrain]"
}
#pretrain_rub2a

pretrain_rub2a_dev(){
  #RBM pretrain
  log_start "rub2a [dev pretrain]"
  dir=exp_multi/rub2a_pretrain/dev_multi
  rbm_init=exp_multi/rub2a_pretrain/final.rbm
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-weight false --learn-visbias false --learn-hidbias true --debug false --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/dev_multi $dir || exit 1;
  log_end "rub2a [dev pretrain]"
}
#pretrain_rub2a_dev


##--------------------
## RBM with utterance dependent visible biases only
pretrain_rub3a(){
  #RBM pretrain
  log_start "rub3a [pretrain]"
  dir=exp_multi/rub3a_pretrain
  rbm_init=exp_multi/dnn1a_pretrain/hid01_rbm/nnet/hid01_rbm_mmt0.9.iter200
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-visbias true --learn-hidbias false --debug true --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/train_multi $dir
  log_end "rub3a [pretrain]"
}
#pretrain_rub3a

pretrain_rub3a_dev(){
  #RBM pretrain
  log_start "rub3a [dev pretrain]"
  dir=exp_multi/rub3a_pretrain/dev_multi
  rbm_init=exp_multi/rub3a_pretrain/final.rbm
  mkdir -p $dir/log
  steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-weight false --learn-visbias true --learn-hidbias false --debug false --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/dev_multi $dir || exit 1;
  log_end "rub3a [dev pretrain]"
}
#pretrain_rub3a_dev






