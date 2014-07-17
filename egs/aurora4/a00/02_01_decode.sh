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

# decoding using oracle LINs
decode_oracle_uttlin(){
  log_start "decode [oracle uttlin]"
  # LIN dir
  lindir=$dir/iter1_lins/test01
  lin_nnet=$dir/iter1_lin.init
  steps/decode_nnet_lin.sh --nj 8 --lin-nnet $lin_nnet --lindir $lindir \
    --srcdir exp_multi/dnn1d \
    exp_multi/dnn1d/graph_bg feat/fbank/test01 $dir/decode_oracle_uttlin/test01
  log_end "decode [oracle uttlin]"
}
#decode_oracle_uttlin

# decoing using SI DNN's recognition results
decode_uttlin(){
  log_start "decode [uttlin]"
  # decoded hypothese
  tra_align_dir=exp_multi/dnn1d
  tra=$tra_align_dir/decode/decode_bg_test01/scoring/17.tra
  # LIN
  lin_nnet=$dir/iter1_lin.init
  steps/decode_nnet_lin.sh --nj 8 --lin-nnet $lin_nnet \
    --tra $tra --tra-align-dir $tra_align_dir \
    --srcdir exp_multi/dnn1d \
    exp_multi/dnn1d/graph_bg feat/fbank/test01 $dir/decode_uttlin/test01
  log_end "decode [uttlin]"
}
decode_uttlin



