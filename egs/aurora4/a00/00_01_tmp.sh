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

train_dnn2b(){
  log_start "dnn2b [train]"
  dir=exp_multi/dnn2b
  dbn=exp_multi/dnn2a_pretrain/hid06c_transf
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.015 \
    --labels ark:exp_multi/dnn2a_pretrain/tri1a/pdf_align/train_dev.pdf \
    --alidir exp_multi/dnn2a_pretrain/tri1a/pdf_align \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  #utils/mkgraph.sh data/lang_bcb05cnp exp_multi/dnn2b exp_multi/dnn2b/graph_bg || exit 1;
  #log_end "dnn2b [train]"
}
#train_dnn2b

train_dnn2c(){
  log_start "dnn2c [train]"
  dir=exp_multi/dnn2c
  ali=exp_multi/tri1a_ali/train_multi
  ali_dev=exp_multi/tri1a_ali/dev_multi
  dbn=exp_multi/dnn2a_pretrain/hid06c_transf
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.015 \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/dnn2c exp_multi/dnn2c/graph_bg || exit 1;
  log_end "dnn2c [train]" 
}
train_dnn2c

