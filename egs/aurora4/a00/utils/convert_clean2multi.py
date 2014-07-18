#!/usr/bin/env python
# 
# This script converts the input clean list to a multi-style list
# by setting the 1st sentence of every 14 sentences to condition 1;
#            the 2nd sentence of every 14 sentences to condition 2;
#							...
#						 the 14th sentence of every 14 sentences to condition 14;
#

import sys, string

if len(sys.argv)!=5:
	print 'Usage: script [map][in][out][utt2nos]'
	sys.exit(1)

mapfile=sys.argv[1]
infile=sys.argv[2]
outfile=sys.argv[3]
utt2nos=sys.argv[4]

tset=[]
nos=[]

fin=open(mapfile)
while True:
	sr=fin.readline()
	if sr=='':break
	tset.append((sr[sr.find(' ')+1:]).strip())
	nos.append('noise'+(sr[:sr.find(' ')]).strip())
fin.close()

flst=[]
fin=open(infile)
while True:
	sr=fin.readline()
	if sr=='':break
	flst.append(sr.strip())
fin.close()

fu2n=open(utt2nos, 'w')
for ii in range(14):
	if tset[ii].endswith('_wv1'):
		for jj in range(ii, len(flst), 14):
			flst[jj]=string.replace(flst[jj], 'clean_wv1', tset[ii])
			sr=flst[jj]
			print >>fu2n, sr[sr.rfind('/')+1:], nos[ii]
	else:
		for jj in range(ii, len(flst), 14):
			flst[jj]=string.replace(flst[jj], 'clean_wv1', tset[ii])
			flst[jj]=string.replace(flst[jj], '.wv1', '.wv2')
			sr=flst[jj]
			print >>fu2n, sr[sr.rfind('/')+1:], nos[ii]
fu2n.close()

fout=open(outfile,'w')
for itm in flst:
	print >>fout, itm
fout.close()

