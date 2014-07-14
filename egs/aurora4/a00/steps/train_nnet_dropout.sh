#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
mlp_init=          # select initialized MLP (override initialization)
#mlp_proto=         # select network prototype (initialize it)
#proto_opts=        # non-default options for 'make_nnet_proto.py'
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
#
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
hid_dim=2048       # select hidden dimension
bn_dim=            # set a value to get a bottleneck network
dbn=               # select DBN to prepend to the MLP initialization
#
init_opts=         # options, passed to the initialization script

# FEATURE PROCESSING
# feature config (applies always)
norm_vars=true # use variance normalization?
# feature_transform:
splice=5         # temporal splicing

# LABELS
labels=            # use these labels to train (override deafault pdf alignments) 
num_tgt=           # force to use number of outputs in the MLP (default is autodetect)
# or can directly provide alignment dir
alidir=
alidir_cv=

# TRAINING SCHEDULER
average_grad=false
bunchsize=128 #size of the Stochastic-GD update block
l1_penalty=0.0
l2_penalty=0.0 #L2 regularization penalty
l2_upperbound=15.0 

momentum_init=0.1
momentum_inc=0.1
momentum_final=0.9
num_iters_momentum_adjust=5

high_learn_rate=0.005
num_iters_high_lrate=30

low_learn_rate=0.001
num_iters_low_lrate=20

#dropout configs
drop_input=false
input_drop_ratio=0.2
hidden_drop_ratio=0.5

train_opts=        # options, passed to the training script
train_tool="nnet-train-xent-hardlab-frmshuff"       # optionally change the training tool

# OTHER
use_gpu_id= # manually select GPU id to run on, (-1 disables GPU)
seed=777    # seed value used for training data shuffling and initialization
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
dir=$4

if [ -z $labels ] && ([ -z $alidir ] || [ -z $alidir_cv ]); then
  echo "No label specified" && exit 1;
fi

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"
if [ ! -z $labels ]; then
  printf "\t Train-set : $data $labels \n"
  printf "\t CV-set    : $data_cv $labels \n"
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels" 
else
  printf "\t Train-set : $data $alidir \n"
  printf "\t CV-set    : $data_cv $alidir_cv \n"
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
  # define pdf-alignment rspecifiers
  labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
  labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- |"
  # 
  labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |" # for analyze-counts.
  labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"

  # get pdf-counts, used later to post-process DNN posteriors
  analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
  # copy the old transition model, will be needed by decoder
  copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1
  # copy the tree
  cp $alidir/tree $dir/tree || exit 1

  # make phone counts for analysis
  analyze-counts --verbose=1 --symbol-table=$lang/phones.txt "$labels_tr_phn" /dev/null 2>$dir/log/analyze_counts_phones.log || exit 1
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

###### PREPARE FEATURE PIPELINE ######

# prepare the features, add delta
feats_tr="ark:add-deltas --delta-order=2 --delta-window=3 scp:$dir/train.scp ark:- |"
feats_cv="ark:add-deltas --delta-order=2 --delta-window=3 scp:$dir/cv.scp ark:- |"

# CMVN:
cmvn_tr=$data/cmvn_0_d_a.utt.scp
cmvn_cv=$data_cv/cmvn_0_d_a.utt.scp
echo "Will use CMVN statistics : ${cmvn_tr}, ${cmvn_cv}"
[ ! -r ${cmvn_tr} ] && echo "Cannot find cmvn stats $cmvn_tr" && exit 1;
[ ! -r ${cmvn_cv} ] && echo "Cannot find cmvn stats $cmvn_cv" && exit 1;
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=$norm_vars scp:$cmvn_tr ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=$norm_vars scp:$cmvn_cv ark:- ark:- |"
# keep track of norm_vars option
echo "$norm_vars" >$dir/norm_vars 

# Splicing:
feats_tr="$feats_tr splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
feats_cv="$feats_cv splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
echo "${splice}" > $dir/splice

# get input dim
echo "Getting input dim : "
num_fea=$(feat-to-dim "$feats_tr" -)
echo "Input dim is : $num_fea"

###### INITIALIZE THE NNET ######
echo 
echo "# NN-INITIALIZATION"
if [ ! -z "$mlp_init" ]; then 
  echo "Using pre-initialized network '$mlp_init'";
else
  [ -z "$dbn" ] && echo "No unsupervised dbn specified!" && exit 1;
  mlp_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  echo "Initializing DNN $mlp_init"
  #initializing the MLP, get the i/o dims...
  #input-dim
  { #optioanlly take output dim of DBN
    [ ! -z $dbn ] && num_fea=$(nnet-forward $dbn "$feats_tr" ark:- | feat-to-dim ark:- -)
    [ -z "$num_fea" ] && echo "Getting nnet input dimension failed!!" && exit 1
  }

  #output-dim
  [ -z $num_tgt ] && num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

  # copy the dbn
  [ -e $mlp_init ] && rm $mlp_init
  [ ! -z $dbn ] && cp $dbn $mlp_init
  layer_config="${num_fea}"
  for i in `seq 1 ${hid_layers}`; do
    layer_config="${layer_config}:${hid_dim}"
  done
  layer_config="${layer_config}:${num_tgt}"
  utils/gen_mlp_dropout_init.py --dim=${layer_config} --gauss --negbias --dropInput=${drop_input} --inputDropRatio=${input_drop_ratio} --hiddenDropRatio=${hidden_drop_ratio} >> $mlp_init
fi

###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING (DROPOUT)"
# stage 1: first 5 iterations of momentum adjusting
steps/finetune_dropout.sh ${feature_transform:+ --feature-transform "$feature_transform"} \
  --num_iters ${num_iters_momentum_adjust} --momentum-init ${momentum_init} --momentum-inc ${momentum_inc} \
  --learn-rate ${high_learn_rate} --bunchsize ${bunchsize} --l1-penalty ${l1_penalty} \
  --l2-penalty ${l2_penalty} --l2-upperbound ${l2_upperbound} --average-grad ${average_grad} \
  "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" \
  $mlp_init $dir/nnet/nnet_stage1

# stage 2: 30 epochs of high learning rate
steps/finetune_dropout.sh ${feature_transform:+ --feature-transform "$feature_transform"} \
  --num_iters ${num_iters_high_lrate} --momentum-init ${momentum_final} --momentum-inc 0.0 \
  --learn-rate ${high_learn_rate} --bunchsize ${bunchsize} --l1-penalty ${l1_penalty} \
  --l2-penalty ${l2_penalty} --l2-upperbound ${l2_upperbound} --average-grad ${average_grad} \
  "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" \
  $dir/nnet/nnet_stage1 $dir/nnet/nnet_stage2

# stage 3: 20 epochs of low learning rate
steps/finetune_dropout.sh ${feature_transform:+ --feature-transform "$feature_transform"} \
  --num_iters ${num_iters_low_lrate} --momentum-init ${momentum_final} --momentum-inc 0.0 \
  --learn-rate ${low_learn_rate} --bunchsize ${bunchsize} --l1-penalty ${l1_penalty} \
  --l2-penalty ${l2_penalty} --l2-upperbound ${l2_upperbound} --average-grad ${average_grad} \
  "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" \
  $dir/nnet/nnet_stage2 $dir/nnet/nnet_stage3
    
# link to the final
mlp_final=`readlink $dir/nnet/nnet_stage3`
[[ ${mlp_final:0:1} != "/" && ${mlp_final:0:1} != "~" ]] && mlp_final=$PWD/$mlp_final
( cd $dir; ln -s $mlp_final final.nnet; )

echo "$0 successfuly finished.. $dir/final.nnet"

sleep 3
exit 0
