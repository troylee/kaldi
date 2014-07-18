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
#pretain_rub1a_test

train_rub1b(){
  #RBM pretrain
  log_start "rub1b [train]"
  dir=exp_multi/rub1b
  mkdir -p $dir/log

  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # prepare the RBM(uttbias) model
  rubdir=exp_multi/rub1a_pretrain
  rbm_mdl=$rubdir/final.rbm
  hidbias=ark:$rubdir/final_hidbias.ark
  hidbias_dev=ark:$rubdir/dev_multi/final_hidbias.ark
  # prepare the init DNN
  sub-nnet --binary=true exp_multi/rbmdnn1a/final.nnet $dir/nnet.init `seq 2 1 15`
  steps/rbmdnn/train_rbmdnn.sh --norm-vars true --mlp-init $dir/nnet.init --learn-rate 0.015 \
    --rbm-mdl $rbm_mdl --hidbias $hidbias --hidbias-cv $hidbias_dev \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rub1b [train]"
}
#train_rub1b

decode_rub1b(){
  log_start "rub1b [decode]"
  dir=exp_multi/rub1b
  rubdir=exp_multi/rub1a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir \
      --rbm-mdl $rubdir/final.rbm --hidbias ark:$rubdir/test/${x}/final_hidbias.ark \
      $dir/graph feat/fbank/${x} $dir/decode/${x} || exit 1;
  done
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub1b [decode]"

}
#decode_rub1b

train_rub1c(){
  #RBM pretrain
  log_start "rub1c [train]"
  dir=exp_multi/rub1c
  mkdir -p $dir/log

  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # prepare the RBM(uttbias) model
  rubdir=exp_multi/rub1a_pretrain
  rbm_mdl=$rubdir/final.rbm
  hidbias=ark:$rubdir/final_hidbias.ark
  hidbias_dev=ark:$rubdir/dev_multi/final_hidbias.ark
  # prepare the init DNN
  sub-nnet --binary=true exp_multi/rbmdnn1a/final.nnet $dir/nnet.init `seq 2 1 15`
  steps/rbmdnn/train_rbmdnn.sh --norm-vars true --mlp-init $dir/nnet.init --learn-rate 0.015 \
    --rbm-mdl $rbm_mdl --hidbias $hidbias --hidbias-cv $hidbias_dev \
    --alidir $ali --alidir-cv $ali_dev --accept-first-update true \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rub1c [train]"
}
#train_rub1c

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

pretain_rub2a_test(){
  #RBM pretrain
  log_start "rub2a [test pretrain]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    dir=exp_multi/rub2a_pretrain/test/$x
    rbm_init=exp_multi/rub2a_pretrain/final.rbm
    mkdir -p $dir/log
    steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-weight false --learn-visbias false --learn-hidbias true --debug false --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/$x $dir || exit 1;
  done
  log_end "rub2a [test pretrain]"
}
#pretain_rub2a_test

train_rub2b(){
  #RBM pretrain
  log_start "rub2b [train]"
  dir=exp_multi/rub2b
  mkdir -p $dir/log

  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # prepare the RBM(uttbias) model
  rubdir=exp_multi/rub2a_pretrain
  rbm_mdl=$rubdir/final.rbm
  hidbias=ark:$rubdir/final_hidbias.ark
  hidbias_dev=ark:$rubdir/dev_multi/final_hidbias.ark
  # prepare the init DNN
  sub-nnet --binary=true exp_multi/rbmdnn1a/final.nnet $dir/nnet.init `seq 2 1 15`
  steps/rbmdnn/train_rbmdnn.sh --norm-vars true --mlp-init $dir/nnet.init --learn-rate 0.015 \
    --rbm-mdl $rbm_mdl --hidbias $hidbias --hidbias-cv $hidbias_dev \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rub2b [train]"
}
#train_rub2b

train_rub2c(){
  #RBM pretrain
  log_start "rub2c [train]"
  dir=exp_multi/rub2c
  mkdir -p $dir/log

  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # prepare the RBM(uttbias) model
  rubdir=exp_multi/rub2a_pretrain
  rbm_mdl=$rubdir/final.rbm
  hidbias=ark:$rubdir/final_hidbias.ark
  hidbias_dev=ark:$rubdir/dev_multi/final_hidbias.ark
  # prepare the init DNN
  sub-nnet --binary=true exp_multi/rbmdnn1a/final.nnet $dir/nnet.init `seq 2 1 15`
  steps/rbmdnn/train_rbmdnn.sh --norm-vars true --mlp-init $dir/nnet.init --learn-rate 0.015 \
    --rbm-mdl $rbm_mdl --hidbias $hidbias --hidbias-cv $hidbias_dev \
    --alidir $ali --alidir-cv $ali_dev --accept-first-update true \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rub2c [train]"
}
#train_rub2c


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

pretain_rub3a_test(){
  #RBM pretrain
  log_start "rub3a [test pretrain]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    dir=exp_multi/rub3a_pretrain/test/$x
    rbm_init=exp_multi/rub3a_pretrain/final.rbm
    mkdir -p $dir/log
    steps/rbmdnn/pretrain_rbm_uttbias.sh --learn-weight false --learn-visbias true --learn-hidbias false --debug false --buffersize 2100 --norm-vars true --splice 5 --init-rbm ${rbm_init} feat/fbank/$x $dir || exit 1;
  done
  log_end "rub3a [test pretrain]"
}
#pretain_rub3a_test

train_rub3b(){
  #RBM pretrain
  log_start "rub3b [train]"
  dir=exp_multi/rub3b
  mkdir -p $dir/log

  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # prepare the RBM(uttbias) model
  rubdir=exp_multi/rub3a_pretrain
  rbm_mdl=$rubdir/final.rbm
  # prepare the init DNN
  sub-nnet --binary=true exp_multi/rbmdnn1a/final.nnet $dir/nnet.init `seq 2 1 15`
  steps/rbmdnn/train_rbmdnn.sh --norm-vars true --mlp-init $dir/nnet.init --learn-rate 0.015 \
    --rbm-mdl $rbm_mdl \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rub3b [train]"
}
#train_rub3b

train_rub3c(){
  #RBM pretrain
  log_start "rub3c [train]"
  dir=exp_multi/rub3c
  mkdir -p $dir/log

  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # prepare the RBM(uttbias) model
  rubdir=exp_multi/rub3a_pretrain
  rbm_mdl=$rubdir/final.rbm
  # prepare the init DNN
  sub-nnet --binary=true exp_multi/rbmdnn1a/final.nnet $dir/nnet.init `seq 2 1 15`
  steps/rbmdnn/train_rbmdnn.sh --norm-vars true --mlp-init $dir/nnet.init --learn-rate 0.015 \
    --rbm-mdl $rbm_mdl \
    --alidir $ali --alidir-cv $ali_dev --accept-first-update true \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rub3c [train]"
}
#train_rub3c

