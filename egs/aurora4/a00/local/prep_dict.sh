#!/bin/bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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

# Call this script from one level above, e.g. from the s3/ directory.  It puts
# its output in data/local/.

# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt

# run this from ../
dir=data/local/dict
mkdir -p $dir


# (1) Get the CMU dictionary
svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
  $dir/cmudict || exit 1;

# can add -r 10966 for strict compatibility.


#(2) Dictionary preparation:


# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# The stresses are removed

# silence phones, one per line.
(echo SIL; echo SPN; echo NSN) > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

# nonsilence phones; on each line is a list of phones that correspond
# really to the same base phone and we only keep the base phone.
cat $dir/cmudict/cmudict.0.7a.symbols | perl -ane 's:\r::; print;' | \
 perl -e 'while(<>){
  chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_"; 
  $phones_of{$1} .= "$_ "; }
  foreach $list (values %phones_of) {print $list . "\n"; } ' | \
  awk '{print $1}' > $dir/nonsilence_phones.txt || exit 1;

# A few extra questions that will be added to those obtained by automatically clustering
# the "real" phones.  Now it is simply silence or not.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | awk '{printf("%s ", $1);} END{printf "\n";}' >> $dir/extra_questions.txt || exit 1;

# Now make a version of the lexicon without the silences. Remove the vowel stress
#Also remove the comments from the cmu lexcion and remove the (1), (2) from words with multiple pronounciations
grep -v ';;;' $dir/cmudict/cmudict.0.7a | \
 perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' | \
 perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die; { 
  print "$w " ; for($n=0;$n<@A;$n++) {  
    if (index($A[$n], '0') > -1 || index($A[$n], '1') > -1 || index($A[$n], '2') > -1) { 
      print substr($A[$n], 0, length($A[$n])-1); print " ";
    } else {print "$A[$n] "}; 
  } print "\n"; } ' > $dir/lexicon1_raw_nosil.txt || exit 1;

# Add to cmudict the silences, noises etc.

(echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - $dir/lexicon1_raw_nosil.txt  > $dir/lexicon2_raw.txt || exit 1;


# lexicon.txt is without the _B, _E, _S, _I markers.
# This is the input to wsj_format_data.sh
cp $dir/lexicon2_raw.txt $dir/lexicon.txt


echo "Dictionary preparation succeeded"

