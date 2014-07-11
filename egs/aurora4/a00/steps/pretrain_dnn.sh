#!/bin/bash
#
# Deep neural network pre-training,using fbank features, 
# per utt CMVN
#

# Begin configuration
stage=
config=
cmd=run.pl

#feature config
norm_vars=true #false:CMN, true:CMVN on fbanks
splice=5   #left- and right-splice value

#mlp size
nn_depth=8    #number of hidden layers
nn_dimhid=2048 #dimension of hidden layers

start_layer=

#global config for trainig
iters_rbm_init=5 #number of iterations with low mmt
iters_rbm=100 #number of iterations with high mmt
iters_rbm_low_lrate=$((2*iters_rbm)) #number of iterations for RBMs with gaussian input

start_halving_inc=0.5
end_halving_inc=0.1
halving_factor=0.5

#parameters for RBM pre-training
rbm_lrate=0.1
rbm_lrate_low=0.001
rbm_momentum=0.5
rbm_momentum_high=0.9
rbm_l2penalty=0.0002

seed=777

# End configuration

echo "$0 $@" # Print the command linke for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: steps/pretrain_dnn.sh <data-dir> <exp-dir>"
  echo "e.g.: steps/pretrain_dnn.sh data/train_multi exp_multi/rbm1a"
  echo "main options (for other, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

data=$1
dir=$2

mkdir -p $dir/{log,nnet}

for f in $data/feats.scp $data/cmvn_0_d_a.utt.scp; do
  [ ! -f $f ] && echo "train_rbm.sh: no such file $f" && exit 1;
done

projpath=/home/svu/g0800132/aurora4
datapath=${projpath}/data/fbank_24/files
exppath=${projpath}/exps/multi_16k/b00_dbn

###### PREPARE FEATURES ######
# shuffle the list
echo "Preparing train lists"
cat ${data}/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
# print the list sizes
wc -l $dir/train.scp 

feats="ark:add-deltas --delta-order=2 --delta-window=3 scp:$dir/train.scp ark:- | apply-cmvn --norm-vars=$norm_vars scp:${data}/cmvn_0_d_a.utt.scp ark:- ark:- |"
# keep track of norm_vars option
echo "$norm_vars" >$dir/norm_vars

#add splicing
feats="$feats splice-feats --left-context=$splice --right-context=$splice ark:- ark:- |"
echo "${splice}" > $dir/splice

# get feature dim
echo "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false "$feats" -)
echo "Feature dim is : $feat_dim"

#get the DNN dimensions
num_fea=$feat_dim
num_hid=$nn_dimhid

[ -z $start_layer ] && start_layer=1
if [ ${start_layer} -gt 1 ]; then
  for depth in $(seq -f '%02g' 1 $((start_layer-1))); do
    TRANSF=$dir/hid${depth}c_transf
  done
fi
###### PERFORM THE PRE-TRAINING ######
for depth in $(seq -f '%02g' $start_layer $nn_depth); do
  echo "%%%%%%% PRE-TRAINING DEPTH $depth"
  RBM=$dir/hid${depth}_rbm/nnet/hid${depth}_rbm
  mkdir -p $(dirname $RBM); mkdir -p $(dirname $RBM)/../log
  echo "Pre-training RBM $RBM "
  #The first RBM needs special treatment, because of Gussian input nodes
  if [ "$depth" == "01" ]; then
    #initialize the RBM with gaussian input
    utils/gen_rbm_init.py --dim=${num_fea}:${num_hid} --gauss --negbias --vistype=gauss --hidtype=bern > $RBM.init
    #pre-train with reduced lrate and more iters
    #a)low momentum
    steps/pretrain_rbm.sh --iters $iters_rbm_init --lrate $rbm_lrate_low --momentum $rbm_momentum --l2-penalty $rbm_l2penalty $RBM.init "$feats" ${RBM}_mmt$rbm_momentum
    #b)high momentum
    steps/pretrain_rbm.sh --iters $iters_rbm_low_lrate --lrate $rbm_lrate_low --momentum $rbm_momentum_high --l2-penalty $rbm_l2penalty ${RBM}_mmt$rbm_momentum "$feats" ${RBM}_mmt${rbm_momentum_high}
  else
    #initialize the RBM
    utils/gen_rbm_init.py --dim=${num_hid}:${num_hid} --gauss --negbias --vistype=bern --hidtype=bern > $RBM.init
    #pre-train (with higher learning rate)
    #a)low momentum
    steps/pretrain_rbm.sh --feature-transform $TRANSF --iters $iters_rbm_init --lrate $rbm_lrate --momentum $rbm_momentum --l2-penalty $rbm_l2penalty $RBM.init "$feats" ${RBM}_mmt$rbm_momentum
    #b)high momentum
    steps/pretrain_rbm.sh --feature-transform $TRANSF --iters $iters_rbm --lrate $rbm_lrate --momentum $rbm_momentum_high --l2-penalty $rbm_l2penalty ${RBM}_mmt$rbm_momentum "$feats" ${RBM}_mmt${rbm_momentum_high}
  fi

  #Compose trasform + RBM + multiclass logistic regression
  echo "Compsing the nnet as transforms"
  NEW_TRANSF=$dir/hid${depth}c_transf
  [ ! -r $TRANSF ] && rm $NEW_TRANSF 2>/dev/null
  [ -r $TRANSF ] && cat $TRANSF > $NEW_TRANSF
  rbm-convert-to-nnet --binary=false ${RBM}_mmt${rbm_momentum_high} - >> $NEW_TRANSF
  # currently the initial weights has no final classification layer

  TRANSF=$NEW_TRANSF
done

echo "Pre-training finished."

echo
echo "%%%% REPORT %%%%"
echo "% RBM pre-training progress"
grep -R progress $dir/nnet
echo 
echo "EOF"

