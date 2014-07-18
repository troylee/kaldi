#!/bin/bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems. 
# - Arnab Ghoshal, 29/05/12

# This is further modified to cater the configuration we used
# - Bo Li (li-bo@outlook.com), 04/06/2014

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level WSJ corpus directory."
  echo "It is assumed that there will be a 'wsj0' and a 'wsj1' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

AURORA=$1
CORPUS=$2

dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/nist_lm
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh # Needed for KALDI_ROOT
export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin

cd $dir

# test id to condition map
echo "01 clean_wv1
02 car_wv1
03 babble_wv1
04 restaurant_wv1
05 street_wv1
06 airport_wv1
07 train_wv1
08 clean_wv2
09 car_wv2
10 babble_wv2
11 restaurant_wv2
12 street_wv2
13 airport_wv2
14 train_wv2" > test_id2name.map

# SI-84 clean training data
cat $AURORA/lists/training_clean_sennh_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > train_clean.flist

# SI-84 multi-condition training data
cat $AURORA/lists/training_multicondition_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > train_multi.flist

#Dev Set
for x in $(seq -f "%02g" 01 14); do
  # Dev-set 1 (330x14 utterances, mainly used for supervised adaptation) 
  cat $AURORA/lists/devtest${x}_0330_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > dev${x}.flist
done
# Dev-set 2, used for DNN training CV
cat $AURORA/lists/devtest01_1206_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > dev_clean.flist
# generate the corresponding multi-style list
$utils/convert_clean2multi.py test_id2name.map dev_clean.flist dev_multi.flist dev_multi.utt2nos

#Test Set
for x in $(seq -f "%02g" 01 14); do
  # test set 1 (339x14 utterances)
  cat $AURORA/lists/test${x}_0330_16k.list \
  | $local/aurora2flist.pl $AURORA | sort -u > test${x}.flist
done

# Finding the transcript files:
find -L $CORPUS -iname '*.dot' > dot_files.flist

# Convert the transcripts into our format (no normalization yet)

# Trans and sph
for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14} ; do  
  $local/flist2scp_wSuffix.pl $x.flist | sort > ${x}_sph.scp
  cat ${x}_sph.scp | awk '{print $1}' \
    | $local/find_transcripts.pl dot_files.flist > ${x}.trans1
done

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14}; do
  cat $x.trans1 | $local/normalize_transcript.pl $noiseword \
    | sort > $x.txt || exit 1;
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14}; do
  awk '{printf("%s sox -B -r 16k -e signed -b 16 -c 1 -t raw %s -t wav - |\n", $1, $2);}' < ${x}_sph.scp \
    > ${x}_wav.scp
done

# Make the utt2spk and spk2utt files.
for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14}; do
  cat ${x}_sph.scp | awk '{print $1}' \
    | perl -ane 'chop; m:^...:; print "$_ $&\n";' > $x.utt2spk
  cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

#in case we want to limit lm's on most frequent words, copy lm training word frequency list
cp $CORPUS/wsj0/doc/lng_modl/vocab/wfl_64.lst $lmdir
chmod u+w $lmdir/*.lst # had weird permissions on source.

# only the 5k closed vocabulary task is used
# copy the 5k vocabulary list
cp $CORPUS/wsj0/doc/lng_modl/vocab/wlist5c.nvp $lmdir
chmod u+w $lmdir/wlist5c.nvp

# use 5k bigram language models
cp $CORPUS/wsj0/doc/lng_modl/base_lm/bcb05cnp.z $lmdir/bcb05cnp.gz || exit 1;
chmod u+w $lmdir/bcb05cnp.gz

# trigram would be: !only closed vocabulary here!
cp $CORPUS/wsj0/doc/lng_modl/base_lm/tcb05cnp.z $lmdir/tcb05cnp.gz || exit 1;
chmod u+w $lmdir/tcb05cnp.gz

if [ ! -f wsj0-train-spkrinfo.txt ] || [ `cat wsj0-train-spkrinfo.txt | wc -l` -ne 134 ]; then
  rm -f wsj0-train-spkrinfo.txt
  echo "Getting wsj0-train-spkrinfo.txt from backup location"
  wget --no-check-certificate https://sourceforge.net/projects/kaldi/files/wsj0-train-spkrinfo.txt
fi

if [ ! -f wsj0-train-spkrinfo.txt ]; then
  echo "Could not get the spkrinfo.txt file from LDC website (moved)?"
  echo "This is possibly omitted from the training disks; couldn't find it." 
  echo "Everything else may have worked; we just may be missing gender info"
  echo "which is only needed for VTLN-related diagnostics anyway."
  exit 1
fi
# Note: wsj0-train-spkrinfo.txt doesn't seem to be on the disks but the
# LDC put it on the web.  Perhaps it was accidentally omitted from the
# disks.  

cat $CORPUS/wsj0/doc/spkrinfo.txt \
    ./wsj0-train-spkrinfo.txt  | \
    perl -ane 'tr/A-Z/a-z/; m/^;/ || print;' | \
    awk '{print $1, $2}' | grep -v -- -- | sort | uniq > spk2gender

echo "Data preparation succeeded"

