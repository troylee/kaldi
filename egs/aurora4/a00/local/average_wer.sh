#!/bin/bash

# Copyright 2014 (Author: Bo Li)
# Apache 2.0

# Begin configuration section.  
wer=true # compute WER stats
ser=false # compute SER stats
min_lmwt=9
max_lmwt=20
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
   echo "Usage: utils/average_wer.sh [options] <decode-dir>"
   echo "... where <decode-dir> is assumed to have the individue wer_*"
   echo "e.g.: steps/average_wer.sh exp/mono/decode_dev*"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --wer <bool>                                     # output WER stats"
   echo "  --ser <bool>                                     # output SER stats"
   echo "  --min_lmwt <int>                                 # minumum LM-weight for lattice rescoring "
   echo "  --max_lmwt <int>                                 # maximum LM-weight for lattice rescoring "
   exit 1;
fi

dir=$1

if $wer ; then
  echo "==========================================================="
  echo "Average WER stats for ${dir}"
  echo "-----------------------------------------------------------"

  for lmwt in `seq $min_lmwt 1 $max_lmwt`; do
    avg=`grep WER ${dir}/wer_${lmwt} | awk '{print $4, $6}' | tr ',' ' ' | awk '{s+=$1; t+=$2} END {print s*100.0/t}'`;
    echo "Avg.WER [lmwt=${lmwt}] $avg %"
  done
  echo "==========================================================="
fi

if $ser ; then
  echo "==========================================================="
  echo "Average WER stats for ${dir}"
  echo "-----------------------------------------------------------"

  for lmwt in `seq $min_lmwt 1 $max_lmwt`; do
    avg=`grep SER ${dir}/wer_${lmwt} | awk '{print $4, $6}' | tr ',' ' ' | awk '{s+=$1; t+=$2} END {print s*100.0/t}'`;
    echo "Avg.SER [lmwt=${lmwt}] $avg %"
  done
  echo "==========================================================="
fi


