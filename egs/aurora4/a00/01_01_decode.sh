#!/bin/bash
# Author: Bo Li (li-bo@outlook.com)
#
# This script is specific to our servers for decoding, which is also included in the 
# main script. 
#
cwd=~/tools/kaldi/egs/aurora4/a00
cd $cwd

numNodes=13 # starts from 0, inclusive
nodes=( compg0 compg11 compg12 compg19 compg20 compg22 compg24 compg50 compg51 compg52 compg53 compg0 compg11 compg12 )

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

decode_rub1b(){
  log_start "rub1b [decode]"
  dir=exp_multi/rub1b
  rubdir=exp_multi/rub1a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir --rbm-mdl $rubdir/final.rbm --hidbias ark:$rubdir/test/${x}/final_hidbias.ark $dir/graph feat/fbank/${x} $dir/decode/${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub1b [decode]"

}
#decode_rub1b

decode_rub1c(){
  log_start "rub1c [decode]"
  dir=exp_multi/rub1c
  rubdir=exp_multi/rub1a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir --rbm-mdl $rubdir/final.rbm --hidbias ark:$rubdir/test/${x}/final_hidbias.ark $dir/graph feat/fbank/${x} $dir/decode/${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub1c [decode]"

}
#decode_rub1c

decode_rub2b(){
  log_start "rub2b [decode]"
  dir=exp_multi/rub2b
  rubdir=exp_multi/rub2a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir --rbm-mdl $rubdir/final.rbm --hidbias ark:$rubdir/test/${x}/final_hidbias.ark $dir/graph feat/fbank/${x} $dir/decode/${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub2b [decode]"

}
#decode_rub2b

decode_rub2c(){
  log_start "rub2c [decode]"
  dir=exp_multi/rub2c
  rubdir=exp_multi/rub2a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir --rbm-mdl $rubdir/final.rbm --hidbias ark:$rubdir/test/${x}/final_hidbias.ark $dir/graph feat/fbank/${x} $dir/decode/${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub2c [decode]"

}
#decode_rub2c

decode_rub3b(){
  log_start "rub3b [decode]"
  dir=exp_multi/rub3b
  rubdir=exp_multi/rub3a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir --rbm-mdl $rubdir/final.rbm $dir/graph feat/fbank/${x} $dir/decode/${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub3b [decode]"

}
#decode_rub3b

decode_rub3c(){
  log_start "rub3c [decode]"
  dir=exp_multi/rub3c
  rubdir=exp_multi/rub3a_pretrain
  inv_acwt=17
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  i=1
  while [ $i -le 14 ]; do
    for j in `seq 0 $numNodes`; do
      sid=$((i+j))
      if [ $sid -le 14 ]; then
        printf -v x 'test%02g' $sid
        echo ${nodes[$j]} $x
        ( ssh ${nodes[$j]} "cd $cwd; steps/rbmdnn/decode_rbmdnn.sh --nj 8 --acwt $acwt --beam 15.0 --latbeam 9.0 --srcdir $dir --rbm-mdl $rubdir/final.rbm $dir/graph feat/fbank/${x} $dir/decode/${x}" ) &
      fi
    done
    wait;
    i=$((sid+1))
  done
  # write out the average WER results
  local/average_wer.sh "$dir/decode/test*" | tee $dir/decode/test.avgwer
  log_end "rub3c [decode]"

}
#decode_rub3c


