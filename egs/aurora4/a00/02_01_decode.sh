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
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_nnet_lin.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --lin-nnet $lin_nnet --lindir $lindir --srcdir exp_multi/dnn1d exp_multi/dnn1d/graph_bg feat/fbank/$x $dir/decode_oracle_uttlin/decode_bg_$x" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh '$dir/decode_oracle_uttlin/decode_bg_test*' | tee $dir/decode_oracle_uttlin/decode_bg_test.avgwer
  log_end "decode [oracle uttlin]"
}
decode_oracle_uttlin

# decoing using SI DNN's recognition results
decode_uttlin(){
  log_start "decode [uttlin]"
  # decoded hypothese
  tra_align_dir=exp_multi/dnn1d
  tra=$tra_align_dir/decode/decode_bg_test01/scoring/17.tra
  # LIN
  lin_nnet=$dir/iter1_lin.init
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_nnet_lin.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --lin-nnet $lin_nnet --tra $tra --tra-align-dir $tra_align_dir --srcdir exp_multi/dnn1d exp_multi/dnn1d/graph_bg feat/fbank/$x $dir/decode_uttlin/decode_bg_$x" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh '$dir/decode_uttlin/decode_bg_test*' | tee $dir/decode_uttlin/decode_bg_test.avgwer
  log_end "decode [uttlin]"
}
#decode_uttlin


