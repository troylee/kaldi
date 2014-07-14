#!/bin/bash
# Copyright 2014 Bo Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# To be run from ..
#
# Restricted Boltzmann Machine training by Contrastive Divergence (CD-1) algorithm.
# Only one layer RBM is trained with utterance dependent biases.
# 

# Begin configuration.
#
# nnet config
hid_dim=2048   #number of units per layer
# number of iterations
init_rbm=  # use a given initial model rather than a randomly generated one
rbm_iters=105             #number of pre-training epochs with high momentum
# pre-training opts
rbm_lrate=0.001        #RBM learning rate
rbm_l2penalty=0.0002  #L2 penalty (increases RBM-mixing rate)
rbm_momentum_change_iter=6  # change momentum at the 6th iteration
rbm_momentum_low=0.5
rbm_momentum_high=0.9
rbm_extra_opts=
# data processing config
# feature config
norm_vars=true     # CMVN or CMN
splice=5           # Temporal splicingl

learn_weight=true
learn_visbias=true
learn_hidbias=true

# misc.
verbose=1 # enable per-cache reports
debug=false
buffersize=1000
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
   echo "Usage: $0 <data> <exp-dir>"
   echo " e.g.: $0 data/train exp/rbm_pretrain"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>           # config containing options"
   echo ""
   echo "  --hid-dim <N>                    # number of hidden units per layer"
   echo "  --rbm-iter <N>                   # number of CD-1 iterations per layer"
   echo "  --rbm-lrate <float>              # learning-rate for Bernoulli-Bernoulli RBMs"
   echo "  --norm-vars <bool>               # use variance normalization (opt.)"
   echo "  --splice <N>                     # splice +/-N frames of input features"
   exit 1;
fi

data=$1
dir=$2

for f in $data/feats.scp $data/cmvn_0_d_a.utt.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo "# INFO"
echo "$0 : Training a RBM with utterance dependent biases"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data \n"

[ -e $dir/final.rbm ] && echo "$0 Skipping, already have $dir/final.rbm" && exit 0

mkdir -p $dir/log

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
# print the list size
wc -l $dir/train.scp

###### PREPARE FEATURE PIPELINE ######

#prepare features, add delta
feats="ark:add-deltas --delta-order=2 --delta-window=3 scp:$dir/train.scp ark:- |"
# do utt-cmvn
# keep track of norm_vars option
echo "$norm_vars" >$dir/norm_vars
cmvn=$data/cmvn_0_d_a.utt.scp
echo "Will use CMVN statistics : $cmvn"
feats="${feats} apply-cmvn --norm-vars=$norm_vars scp:$cmvn ark:- ark:- |"
# splicing
# keep track of splice option
echo "$splice" > $dir/splice
feats="${feats} splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"

#get feature dim
echo -n "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false scp:$dir/train.scp -)
echo $feat_dim

###### GET THE DIMENSIONS ######
num_fea=$(feat-to-dim --print-args=false "${feats}" -)
num_hid=$hid_dim

###### PERFORM THE PRE-TRAINING ######
echo
echo "# PRE-TRAINING RBM"
RBM=$dir/rbm_uttbias
visbias=$dir/$depth.rbm_visbias.ark
hidbias=$dir/$depth.rbm_hidbias.ark

#The first RBM needs special treatment, because of Gussian input nodes
if [ -z "$init_rbm" ]; then
  #This is Gaussian-Bernoulli RBM
  #initialize
  echo "Initializing '$RBM.init'"
  utils/gen_rbm_init.py --dim=${num_fea}:${num_hid} --gauss --negbias --vistype=gauss --hidtype=bern > $RBM.init
  init_rbm=$RBM.init
fi

# low momentum
cur_mdl=${init_rbm}
momentum=$rbm_momentum_low
echo "RBM training:"
for iter in $(seq 1 $rbm_iters); do
  printf "Iteration ${iter}/${rbm_iters}: "
  new_mdl=${RBM}_iter${iter}

  if [ $iter -eq ${rbm_momentum_change_iter} ]; then
    momentum=$rbm_momentum_high
  fi

  if [ -e $dir/.done_iter${iter} ] ; then
    printf "skipping ... \n" 
    $learn_weight && cur_mdl=$new_mdl ;
  else
    bias_opt=""
    if $learn_visbias ; then
      if [ $iter -gt 1 ]; then
        bias_opt="${bias_opt} --visbias-in=ark:$dir/visbias_iter$((iter-1)).ark"
      fi
      bias_opt="${bias_opt} --visbias-out=ark:$dir/visbias_iter${iter}.ark"
    fi
    if $learn_hidbias ; then
      if [ $iter -gt 1 ]; then
        bias_opt="${bias_opt} --hidbias-in=ark:$dir/hidbias_iter$((iter-1)).ark"
      fi
      bias_opt="${bias_opt} --hidbias-out=ark:$dir/hidbias_iter${iter}.ark"
    fi
    if $learn_weight ; then
      [ ! -e ${cur_mdl} ] && echo "${cur_mdl} not found!" && exit 1;
      rbm-uttbias-train --binary=true --verbose=$verbose --buffer-size=$buffersize \
        $bias_opt $rbm_extra_opts \
        --momentum=$momentum --learn-rate=$rbm_lrate --l2-penalty=$rbm_l2penalty \
        "$feats" ${cur_mdl} ${new_mdl} 2>$dir/log/rbm_tr_iter${iter}.log || exit 1

      (( ! $debug )) && [ ${iter} -gt 1 ] && rm ${cur_mdl} ;
      # update the current model 
      cur_mdl=$new_mdl
    else
      rbm-uttbias-train --binary=true --verbose=$verbose --buffer-size=$buffersize \
        $bias_opt $rbm_extra_opts \
        --momentum=$momentum --learn-rate=$rbm_lrate --l2-penalty=$rbm_l2penalty \
        "$feats" ${init_rbm} 2>$dir/log/rbm_tr_iter${iter}.log || exit 1
    fi

    # clean up the intermediate bias estimates
    if ! $debug && [ ${iter} -gt 1 ]; then
      $learn_visbias && rm $dir/visbias_iter$((iter-1)).ark
      $learn_hidbias && rm $dir/hidbias_iter$((iter-1)).ark
    fi
    touch $dir/.done_iter${iter}
    printf "done\n"
  fi
  
done

# link the final rbm
$learn_weight && ( cd $dir; ln -s rbm_uttbias_iter${iter} final.rbm; )
$learn_visbias && ( cd $dir; ln -s visbias_iter${iter}.ark final_visbias.ark; )
$learn_hidbias && (cd $dir; ln -s hidbias_iter${iter}.ark final_hidbias.ark; )

echo "RBM training finished."

sleep 3
exit 0

