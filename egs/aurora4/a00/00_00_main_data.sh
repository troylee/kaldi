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
# data preparations
#

aurora4=/media/research/corpus/aurora4
#we need lm, trans, from WSJ0 CORPUS
wsj0=/media/research/corpus/WSJ0

prepare_basic(){
  log_start "data preparation"
  local/prep_data.sh $aurora4 $wsj0
  log_end "data preparation"

  log_start "dictionary preparation"
  local/prep_dict.sh || exit 1;
  log_end "dictionary preparation"

  log_start "lang preparation"
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
  log_end "lang preparation"

  log_start "format data"
  local/prep_testlm.sh || exit 1;
  log_end "format_data"

  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  log_start "MFCC extraction"
  mfccdir=feat/mfcc
  mkdir -p $mfccdir
  for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14} ; do 
    if [ -d $mfccdir/$x ]; then
      rm -r $mfccdir/$x
    fi
    cp -r data/$x $mfccdir/$x
    steps/make_mfcc.sh --nj 1 $mfccdir/$x exp/make_mfcc/$x $mfccdir/params || exit 1;
    steps/compute_cmvn_utt_stats.sh $mfccdir/$x exp/make_mfcc/$x $mfccdir/params || exit 1;
  done
  log_end "MFCC extraction"

  # make fbank features
  log_start "FBank extraction"
  fbankdir=feat/fbank
  mkdir -p $fbankdir
  for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14} ; do
    if [ -d $fbankdir/$x ]; then
      rm -r $fbankdir/$x
    fi
    cp -r data/$x $fbankdir/$x
    steps/make_fbank.sh --nj 1 $fbankdir/$x exp/make_fbank/$x $fbankdir/params || exit 1;
    steps/compute_cmvn_utt_stats.sh $fbankdir/$x exp/make_fbank/$x $fbankdir/params || exit 1;
  done
  log_end "FBank extraction"
}
prepare_basic


