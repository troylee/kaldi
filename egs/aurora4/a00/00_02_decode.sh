#!/bin/bash
# Author: Bo Li (li-bo@outlook.com)
#
# This script is specific to our servers for decoding, which is also included in the 
# main script. 
#
cwd=~/tools/kaldi/egs/aurora4/a00
cd $cwd

numNodes=13 # starts from 0, inclusive
nodes=( compg0 compg11 compg12 compg19 compg20 compg22 compg24 compg50 compg51 compg52 compg53 compg51 compg52 compg53 )

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

decode_clean_tri1a(){
  # decode exp_clean/tri1a
  log_start "tri1a [decode]"
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_deltas.sh --nj 8 --srcdir exp_clean/tri1a exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1a/decode/decode_bg_test*' | tee exp_clean/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_clean_tri1a

decode_clean_tri1b(){
  log_start "tri1b [decode]"
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_deltas.sh --nj 8 --srcdir exp_clean/tri1b exp_clean/tri1b/graph_bg feat/mfcc/${x} exp_clean/tri1b/decode/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1b/decode/decode_bg_test*' | tee exp_clean/tri1b/decode/decode_bg_test.avgwer
  log_end "tri1b [decode]"
}
#decode_clean_tri1b

decode_clean_tri1b_vtsmodel(){
  log_start "tri1b [vtsmodel decode]"
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_vts_model.sh --nj 8 --srcdir exp_clean/tri1b exp_clean/tri1b/graph_bg feat/mfcc/${x} exp_clean/tri1b/decode_vts_model/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1b/decode_vts_model/decode_bg_test*' | tee exp_clean/tri1b/decode_vts_model/decode_bg_test.avgwer
  log_end "tri1b [vtsmodel decode]"
}
#decode_clean_tri1b_vtsmodel


decode_multi_tri1a(){
  # decode exp_multi/tri1a
  log_start "tri1a [decode]"
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_deltas.sh --nj 8 --srcdir exp_multi/tri1a exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1a/decode/decode_bg_test*' | tee exp_multi/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_multi_tri1a

decode_multi_tri1b(){
  log_start "tri1b [decode]"
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_deltas.sh --nj 8 --srcdir exp_multi/tri1b exp_multi/tri1b/graph_bg feat/mfcc/${x} exp_multi/tri1b/decode/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1b/decode/decode_bg_test*' | tee exp_multi/tri1b/decode/decode_bg_test.avgwer
  log_end "tri1b [decode]"
}
#decode_multi_tri1b

decode_multi_tri1b_vtsmodel(){
  log_start "tri1b [vtsmodel decode]"
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_vts_model.sh --nj 8 --srcdir exp_multi/tri1b exp_multi/tri1b/graph_bg feat/mfcc/${x} exp_multi/tri1b/decode_vts_model/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1b/decode_vts_model/decode_bg_test*' | tee exp_multi/tri1b/decode_vts_model/decode_bg_test.avgwer
  log_end "tri1b [vtsmodel decode]"
}
#decode_multi_tri1b_vtsmodel

## dnn2b is the old RBMs, old tri1a and new features and fine-tuning
decode_multi_dnn2b(){
  log_start "dnn2b [decode]"
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_nnet.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir exp_multi/dnn2b --model exp_multi/dnn2a_pretrain/tri1a/pdf_align/final.mdl --class-frame-counts exp_multi/dnn2a_pretrain/tri1a/pdf_align/train.counts exp_multi/dnn2a_pretrain/tri1a/graph_bcb05cnp feat/fbank/${x} exp_multi/dnn2b/decode/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/dnn2b/decode/decode_bg_test*' | tee exp_multi/dnn2b/decode/decode_bg_test.avgwer
  log_end "dnn2b [decode]"
}
#decode_multi_dnn2b

## dnn2c is using the old RBMs, new tri1a, new features and new fine-tuning
decode_multi_dnn2c(){
  log_start "dnn2c [decode]"
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/decode_nnet.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir exp_multi/dnn2c exp_multi/dnn2c/graph_bg feat/fbank/${x} exp_multi/dnn2c/decode/decode_bg_${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/dnn2c/decode/decode_bg_test*' | tee exp_multi/dnn2c/decode/decode_bg_test.avgwer
  log_end "dnn2c [decode]"
}
#decode_multi_dnn2c

