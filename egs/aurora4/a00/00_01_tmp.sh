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
  log_end "dnn2b [train]"
}
#train_dnn2b

decode_dnn2b(){
  log_start "dnn2b [decode]"
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/decode_nnet.sh --stage 1 --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir exp_multi/dnn2b \
      --model exp_multi/dnn2a_pretrain/tri1a/pdf_align/final.mdl \
      --class-frame-counts exp_multi/dnn2a_pretrain/tri1a/pdf_align/train.counts \
      exp_multi/dnn2a_pretrain/tri1a/graph_bcb05cnp feat/fbank/${x} exp_multi/dnn2b/decode/decode_bg_${x} || exit 1;
    exit 0;
  done
  local/average_wer.sh 'exp_multi/dnn2b/decode/decode_bg_test*' | tee exp_multi/dnn2b/decode/decode_bg_test.avgwer
  log_end "dnn2b [decode]"

}
decode_dnn2b


