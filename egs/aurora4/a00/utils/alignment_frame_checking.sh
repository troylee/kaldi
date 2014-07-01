#!/bin/bash
# Copyright 2014 Bo Li

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 <clean-dir> <multi-dir>"
   echo " e.g.: $0 exp_multi/tri1a_ali/train_clean exp_multi/tri1a_ali/train_clean"
   exit 1;
fi

cleandir=$1
multidir=$2

gunzip -c ${cleandir}/ali.*.gz | awk '{print $1, NF}' | sort > ${cleandir}/alignment_frame_info
gunzip -c ${multidir}/ali.*.gz | awk '{print $1, NF}' | sort > ${multidir}/alignment_frame_info

res=`diff ${cleandir}/alignment_frame_info ${multidir}/alignment_frame_info | wc -l | awk '{print $1}'`

if [ "${res}" == "0" ]; then
  echo "Consistent!"
else
  echo "Warninig! The two alignments seem not matching!"
fi

