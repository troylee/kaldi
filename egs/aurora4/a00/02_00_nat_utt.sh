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

# current exp dir
dir=exp_multi/dnn1d_nat_utt

# initialize the NAT exp
iter1_init(){
  log_start "nat [1 init]"
  mkdir -p $dir/log
  # SI nnet dir for the 1st iteration
  sidir=exp_multi/dnn1d
  # convert the SI nnet to LIN nnet
  lin-init --binary=true --lin-type=0 ${sidir}/final.nnet $dir/iter1_lin.init
  # keep a copy of LIN layer only, as weights and biases for LIN are saved in 
  # separate archives, not with the nnet, the lin.nnet will be used for all 
  # subsequent iterations
  sub-nnet --binary=true $dir/iter1_lin.init $dir/lin.nnet 0
  log_end "nat [1 init]"
}
#iter1_init

# training per-utterance LIN using transcriptions
iter1_train_lins(){
  log_start "nat [1 lins]"
  mkdir -p $dir/iter1_lins
  # SI nnet dir for the 1st iteration
  sidir=exp_multi/dnn1d
  # train data
  steps/train_uttlin_1iter.sh --nj 8 --nnet-lin $dir/iter1_lin.init --srcdir $sidir \
    --alidir exp_multi/dnn1c_ali/train_multi \
    feat/fbank/train_multi data/lang $dir/iter1_lins/train_multi || exit 1;
  # dev data
  steps/train_uttlin_1iter.sh --nj 8 --nnet-lin $dir/iter1_lin.init --srcdir $sidir \
    --alidir exp_multi/dnn1c_ali/dev_multi \
    feat/fbank/dev_multi data/lang $dir/iter1_lins/dev_multi || exit 1;
}
#iter1_train_lins

# estimate test LINs using correct transcriptions for oracle experiments
iter1_train_oracle_uttlin(){
  sidir=exp_multi/dnn1d
  # test data
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/train_uttlin_1iter.sh --learn-rate 0.05 --nj 8 --nnet-lin $dir/iter1_lin.init --srcdir $sidir \
      --alidir exp_multi/dnn1c_ali/$x \
      feat/fbank/$x data/lang $dir/iter1_lins/$x || exit 1;
  done
  log_end "nat [1 lins]"
}
iter1_train_oracle_uttlin

# decoding using oracle LINs
decode_oracle_uttlin(){
  log_start "decode [oracle uttlin]"
  lin_nnet=$dir/iter1_lin.init
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    lindir=$dir/iter1_lins/$x
    steps/decode_nnet_lin.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --lin-nnet $lin_nnet --lindir $lindir --srcdir exp_multi/dnn1d exp_multi/dnn1d/graph feat/fbank/$x $dir/decode_oracle_uttlin/$x
  done
  # write out the average WER results
  local/average_wer.sh '$dir/decode_oracle_uttlin/test*' | tee $dir/decode_oracle_uttlin/test.avgwer
  log_end "decode [oracle uttlin]"
}
decode_oracle_uttlin

# training the nnet using estimated lins
iter1_train_nnet(){
  log_start "nat [1 nnet]"
  mkdir -p $dir/iter1_nnet/log
  # alignments 
  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  # LIN dirs
  lindir=$dir/iter1_lins/train_multi
  lindir_dev=$dir/iter1_lins/dev_multi
  # split the nnet into LIN and standard nnet
  mlp_init=exp_multi/dnn1d/final.nnet
  steps/train_uttlin_nnet.sh --norm-vars true --learn-rate 0.015 \
    --mlp-init ${mlp_init} --lin-nnet $dir/lin.nnet \
    --alidir $ali --alidir-cv $ali_dev \
    --lindir $lindir --lindir-cv $lindir_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir/iter1_nnet || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir/iter1_nnet $dir/iter1_nnet/graph || exit 1;
  log_end "nat [1 nnet]"
}
#iter1_train_nnet

