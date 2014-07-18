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

#------------------------------------------------------
# Noise Adaptive System
# 1. training 14 LINs on training set (maybe iteratively)
# 2. they are used as bases for testing
# 3. each testing utterance estimates the coefficients for those bases
#------------------------------------------------------

# split the training into 14 noise subsets for training of the noise dependent LINS
split_data(){
  log_start "prepare data"
  featdir=feat/fbank/train_multi
  dir=exp_multi/dnn1d_nat_nos
  # spliting file
  sptfile=conf/train_multi.nos2utt
  for i in `seq -f %02g 1 14`; do 
    x=train$i
    echo $x
    mkdir -p $dir/feat/$x
    grep $x $sptfile | sed "s:noise$i ::g" | tr ' ' '\n' > $dir/tmp 
    for f in cmvn_0_d_a.utt.scp feats.scp utt2spk text wav.scp ; do
      [ -f $featdir/$f ] && utils/filter_scp.pl $dir/tmp $featdir/$f > $dir/feat/$x/$f || exit 1;
    done
    [ -f $dir/feat/$x/utt2spk ] && utils/utt2spk_to_spk2utt.pl $dir/feat/$x/utt2spk > $dir/feat/$x/spk2utt || exit 1;
  done
  [ -f $dir/tmp ] && rm $dir/tmp || exit 1;
  log_end "prepare data"
}
#split_data

iter1_prepare(){
  log_start "iter1 [pre]"
  dir=exp_multi/dnn1d_nat_nos
  # SI nnet
  sidir=exp_multi/dnn1d
  # convert the SI nnet to LIN nnet
  lin-init --binary=true --lin-type=0 ${sidir}/final.nnet $dir/iter1_lin_nnet.init
  # keep a copy of LIN layer only, as weights and biases for LIN are saved in 
  # separate archives, not with the nnet, the lin.init will be used for all 
  # subsequent iterations
  sub-nnet --binary=true $dir/iter1_lin_nnet.init $dir/lin.init 0
  log_end "iter1 [pre]"
}
#iter1_prepare

iter1_train_lin(){
  log_start "iter1 [train lin]"
  ali=exp_multi/dnn1c_ali/train_multi
  expdir=exp_multi/dnn1d_nat_nos
  dir=$expdir/iter1_lin
  mlp_init=$expdir/iter1_lin_nnet.init
  # train LIN separately
  for i in `seq -f %02g 1 14`; do 
    x=train$i
    mkdir -p $dir/$x/log
    ali_dev=exp_multi/dnn1c_ali/dev$i
    steps/train_nnet.sh --learn-rate 0.005 --end-learn-rate 0.000001 --norm-vars true --splice 5 \
      --learn-factors "1,0,0,0,0,0,0,0" --accept-first-update true \
      --mlp-init $mlp_init --alidir $ali --alidir-cv $ali_dev \
      $expdir/feat/$x feat/fbank/dev$i data/lang $dir/$x 
  done
  # merge LINs into one single archive
  seq -f '%02g' 1 14 | awk -v pt=${dir} '{print "noise" $1, pt "/" $1 "/lin_final.nnet"}' > ${dir}/lin_xform.ark
  lin-merge ark:$dir/lin_xform.ark ark:$dir/lin_weight.ark ark:$dir/lin_bias.ark
  log_end "iter1 [train lin]"
}
iter1_train_lin


