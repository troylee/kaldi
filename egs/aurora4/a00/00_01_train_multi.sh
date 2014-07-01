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
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1a exp_multi/tri1a/graph_bg || exit 1;
  log_end "tri1a [train]"
}
#train_multi_tri1a

decode_multi_tri1a(){
  log_start "tri1a [decode]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/decode_deltas.sh --nj 4 --srcdir exp_multi/tri1a exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x} || exit 1;
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1a/decode/decode_bg_test*' | tee exp_multi/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_multi_tri1a

align_multi_tri1a(){
  # align multi-style data with multi-trained model, needs a larger beam
  log_start "tri1a [align-train-multi]"
  steps/align_si.sh --nj 4 --retry-beam 60 feat/mfcc/train_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_multi || exit 1;
  log_end "tri1a [align-train-multi]"

  log_start "tri1a [align-dev-multi]"
  steps/align_si.sh --nj 4 --retry-beam 80 feat/mfcc/dev_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_multi || exit 1;
  log_end "tri1a [align-dev-multi]"

  # align clean data with multi-trained model
  log_start "tri1a [align-train-clean]"
  steps/align_si.sh --nj 4 feat/mfcc/train_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_clean || exit 1;
  log_end "tri1a [align-train-clean]"

  log_start "tri1a [align-dev-clean]"
  steps/align_si.sh --nj 4 feat/mfcc/dev_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_clean || exit 1;
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
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1b exp_multi/tri1b/graph_bg || exit 1;
  log_end "tri1b [train]"
}
#train_multi_spr_tri1b

