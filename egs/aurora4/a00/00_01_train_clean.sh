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
# Model training - clean
#

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]

train_clean_tri1a(){
  log_start "mono [train]"
  steps/train_mono.sh --boost-silence 1.25 --nj 4 --norm_vars true \
    feat/mfcc/train_clean data/lang exp_clean/mono || exit 1;
  log_end "mono [train]"

  log_start "mono [align]"
  steps/align_si.sh --boost-silence 1.25 --nj 4  \
     feat/mfcc/train_clean data/lang exp_clean/mono exp_clean/mono_ali || exit 1;
  log_end "mono [align]"

  log_start "tri1a [train]"  
  steps/train_deltas.sh --boost-silence 1.25 --norm_vars true \
      4200 55000 feat/mfcc/train_clean data/lang exp_clean/mono_ali exp_clean/tri1a || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_clean/tri1a exp_clean/tri1a/graph_bg || exit 1;
  log_end "tri1a [train]"
}
#train_clean_tri1a

decode_clean_tri1a(){
  log_start "tri1a [decode]"
  # some system works well will {01..14}, but some will remove the starting 0s.
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/decode_deltas.sh --nj 4 --srcdir exp_clean/tri1a exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode/decode_bg_${x} || exit 1;
  done
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1a/decode/decode_bg_test*' | tee exp_clean/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_clean_tri1a

align_clean_tri1a(){
  # align clean data with clean-trained model
  log_start "tri1a [align-train]"
  steps/align_si.sh --nj 4 --retry-beam 200 feat/mfcc/train_clean data/lang exp_clean/tri1a exp_clean/tri1a_ali/train_clean || exit 1;
  log_end "tri1a [align-train]"
}
align_clean_tri1a

train_clean_spr_tri1b(){
  log_start "tri1b [train]"
  steps/singlepass_retrain.sh feat/mfcc/train_clean exp_clean/tri1a_ali/train_clean exp_clean/tri1b || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_clean/tri1b exp_clean/tri1b/graph_bg || exit 1;
  log_end "tri1b [train]"
}
train_clean_spr_tri1b

