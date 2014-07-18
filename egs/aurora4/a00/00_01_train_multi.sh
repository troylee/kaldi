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

###########################################
# Model training - multi
#

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]

train_multi_tri1a(){
  log_start "mono [train]"
  steps/train_mono.sh --boost-silence 1.25 --nj 4 --norm_vars true \
    feat/mfcc/train_multi data/lang exp_multi/mono || exit 1;
  log_end "mono [train]"

  log_start "mono [align]"
  steps/align_si.sh --boost-silence 1.25 --nj 4  \
     feat/mfcc/train_multi data/lang exp_multi/mono exp_multi/mono_ali || exit 1;
  log_end "mono [align]"

  log_start "tri1a [train]"
  steps/train_deltas.sh --boost-silence 1.25 --norm_vars true \
      4200 55000 feat/mfcc/train_multi data/lang exp_multi/mono_ali exp_multi/tri1a || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1a exp_multi/tri1a/graph || exit 1;
  log_end "tri1a [train]"
}
#train_multi_tri1a

decode_multi_tri1a(){
  log_start "tri1a [decode]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/decode_deltas.sh --nj 4 --srcdir exp_multi/tri1a exp_multi/tri1a/graph feat/mfcc/${x} exp_multi/tri1a/decode/${x} || exit 1;
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1a/decode/test*' | tee exp_multi/tri1a/decode/test.avgwer
  log_end "tri1a [decode]"
}
#decode_multi_tri1a

decode_multi_tri1a_fmllr(){
  log_start "tri1a [fmllr decode]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/decode_deltas_fmllr.sh --nj 8 --srcdir exp_multi/tri1a --si-dir exp_multi/tri1a/decode/${x} exp_multi/tri1a/graph feat/mfcc/${x} exp_multi/tri1a/decode_fmllr/${x} || exit 1;
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1a/decode_fmllr/test*' | tee exp_multi/tri1a/decode_fmllr/test.avgwer
  log_end "tri1a [fmllr decode]"
}
#decode_multi_tri1a_fmllr

align_multi_tri1a(){
  # align multi-style data with multi-trained model, needs a larger beam
  log_start "tri1a [align-train-multi]"
  steps/align_si.sh --nj 4 --retry-beam 200 feat/mfcc/train_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_multi || exit 1;
  log_end "tri1a [align-train-multi]"

  log_start "tri1a [align-dev-multi]"
  steps/align_si.sh --nj 4 --retry-beam 200 feat/mfcc/dev_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_multi || exit 1;
  log_end "tri1a [align-dev-multi]"

  # align clean data with multi-trained model
  log_start "tri1a [align-train-clean]"
  steps/align_si.sh --nj 4 --retry-beam 200 feat/mfcc/train_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_clean || exit 1;
  log_end "tri1a [align-train-clean]"

  log_start "tri1a [align-dev-clean]"
  steps/align_si.sh --nj 4 --retry-beam 200 feat/mfcc/dev_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_clean || exit 1;
  log_end "tri1a [align-dev-clean]"

  # additional processing of the clean data alignments for used as multi labels
  dir=exp_multi/tri1a_ali/train_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do 
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/train_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  dir=exp_multi/tri1a_ali/dev_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do 
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/dev_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  # sanity check for the genreated clean frame alignment
  ./utils/alignment_frame_checking.sh exp_multi/tri1a_ali/train_clean/ exp_multi/tri1a_ali/train_multi/
  ./utils/alignment_frame_checking.sh exp_multi/tri1a_ali/dev_clean/ exp_multi/tri1a_ali/dev_multi/
}
#align_multi_tri1a

train_multi_spr_tri1b(){
  log_start "tri1b [train]"
  steps/singlepass_retrain.sh feat/mfcc/train_multi exp_multi/tri1a_ali/train_multi exp_multi/tri1b || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1b exp_multi/tri1b/graph || exit 1;
  log_end "tri1b [train]"
}
#train_multi_spr_tri1b

###############################################
#Verifying old DNN setup, using the existing RBM
# for self-check only

prepare_dnn2a_pretrain(){
  log_start "dnn2a [pretrain]"
  dir=exp_multi/dnn2a_pretrain
  srcdir=/media/research/dstc_aurora4
  mkdir $dir
  cp $srcdir/exps/multi_16k/b00_dnn/nnet/hid06c_transf $dir
  ln -s $srcdir/exps/multi_16k/a00_hmm/tri1a $dir/tri1a
  cp $srcdir/lib/fsts/wsj_5cnp/words.txt $dir/tri1a/graph_bcb05cnp/words.txt
  log_end "dnn2a [pretrain]"
}
#prepare_dnn2a_pretrain 

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
      exp_multi/dnn2a_pretrain/tri1a/graph_bcb05cnp feat/fbank/${x} exp_multi/dnn2b/decode/${x} || exit 1;
    exit 0;
  done
  local/average_wer.sh 'exp_multi/dnn2b/decode/test*' | tee exp_multi/dnn2b/decode/test.avgwer
  log_end "dnn2b [decode]"

}
#decode_dnn2b

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
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/dnn2c exp_multi/dnn2c/graph || exit 1;
  log_end "dnn2c [train]" 
}
#train_dnn2c


###############################################
#Now begin train DNN systems on multi data

pretrain_dnn1a(){
  #RBM pretrain
  log_start "dnn1a [pretrain]"
  dir=exp_multi/dnn1a_pretrain
  mkdir -p $dir/log
  steps/pretrain_dnn.sh --nn-depth 7 --norm-vars true --splice 5 feat/fbank/train_multi $dir
  log_end "dnn1a [pretrain]"
}
#pretrain_dnn1a

train_dnn1b(){
  log_start "dnn1b [train]"
  dir=exp_multi/dnn1b
  ali=exp_multi/tri1a_ali/train_multi
  ali_dev=exp_multi/tri1a_ali/dev_multi
  dbn=exp_multi/dnn1a_pretrain/hid06c_transf
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.015 \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "dnn1b [train]" 
}
#train_dnn1b

align_dnn1b(){
  #nnet realignments
  log_start "dnn1b [realign-train_multi]"
  steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/train_multi data/lang exp_multi/dnn1b exp_multi/dnn1b_ali/train_multi || exit 1;
  log_end "dnn1b [realign-train_multi]"

  log_start "dnn1b [realign-dev_multi]"
  steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/dev_multi data/lang exp_multi/dnn1b exp_multi/dnn1b_ali/dev_multi || exit 1;
  log_end "dnn1b [realign-dev_multi]"
}
#align_dnn1b

train_dnn1c(){
  log_start "dnn1c [train]"
  dir=exp_multi/dnn1c
  ali=exp_multi/dnn1b_ali/train_multi
  ali_dev=exp_multi/dnn1b_ali/dev_multi
  mlp_init=exp_multi/dnn1b/nnet.init
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --mlp-init ${mlp_init} --learn-rate 0.015 \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "dnn1c [train]" 
}
#train_dnn1c

align_dnn1c(){
  #nnet realignments
  log_start "dnn1c [realign-train_multi]"
  steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/train_multi data/lang exp_multi/dnn1c exp_multi/dnn1c_ali/train_multi || exit 1;
  log_end "dnn1c [realign-train_multi]"

  log_start "dnn1c [realign-dev_multi]"
  steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/dev_multi data/lang exp_multi/dnn1c exp_multi/dnn1c_ali/dev_multi || exit 1;
  log_end "dnn1c [realign-dev_multi]"
}
#align_dnn1c

align_dnn1c_test(){
  log_start "dnn1c [realign-test]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/$x data/lang exp_multi/dnn1c exp_multi/dnn1c_ali/$x || exit 1;
  done
  log_end "dnn1c [realign-test]"
}
#align_dnn1c_test

align_dnn1c_dev(){
  log_start "dnn1c [realign-dev]"
  for i in `seq -f "%02g" 1 14`; do
    x=dev${i}
    steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/$x data/lang exp_multi/dnn1c exp_multi/dnn1c_ali/$x || exit 1;
  done
  log_end "dnn1c [realign-dev]"
}
align_dnn1c_dev

train_dnn1d(){
  log_start "dnn1d [train]"
  dir=exp_multi/dnn1d
  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  mlp_init=exp_multi/dnn1b/nnet.init
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --mlp-init ${mlp_init} --learn-rate 0.015 \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "dnn1d [train]" 
}
#train_dnn1d

align_dnn1d(){
  #nnet realignments
  log_start "dnn1d [realign-train_multi]"
  steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/train_multi data/lang exp_multi/dnn1d exp_multi/dnn1d_ali/train_multi || exit 1;
  log_end "dnn1d [realign-train_multi]"

  log_start "dnn1d [realign-dev_multi]"
  steps/align_nnet.sh --nj 4 --retry-beam 200 feat/fbank/dev_multi data/lang exp_multi/dnn1d exp_multi/dnn1d_ali/dev_multi || exit 1;
  log_end "dnn1d [realign-dev_multi]"
}
#align_dnn1d

train_dnn1e(){
  log_start "dnn1e [train]"
  dir=exp_multi/dnn1e
  ali=exp_multi/dnn1d_ali/train_multi
  ali_dev=exp_multi/dnn1d_ali/dev_multi
  mlp_init=exp_multi/dnn1b/nnet.init
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --mlp-init ${mlp_init} --learn-rate 0.015 \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "dnn1e [train]" 
}
#train_dnn1e

#--------------------
# 7 hidden layers
train_dnn1d_7h(){
  log_start "dnn1d_7h [train]"
  dir=exp_multi/dnn1d_7h
  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  dbn=exp_multi/dnn1a_pretrain/hid07c_transf
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.015 \
    --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "dnn1d_7h [train]"
}
#train_dnn1d_7h

#--------------------
# RBM-DNN h07r01
train_rbmdnn1a(){
  log_start "rbmdnn1a [train]"
  dir=exp_multi/rbmdnn1a
  ali=exp_multi/dnn1c_ali/train_multi
  ali_dev=exp_multi/dnn1c_ali/dev_multi
  mlp_init=exp_multi/dnn1d_7h/nnet.init
  mkdir -p $dir/log
  steps/train_nnet.sh --norm-vars true --mlp-init ${mlp_init} --learn-rate 0.015 \
    --learn-factors "0,1,1,1,1,1,1,1" --alidir $ali --alidir-cv $ali_dev \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph || exit 1;
  log_end "rbmdnn1a [train]"
}
#train_rbmdnn1a


