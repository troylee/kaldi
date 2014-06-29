#!/usr/bin/env python
# 
# This script converts the input clean list to a multi-style list
# by setting the 1st sentence of every 14 sentences to condition 1;
#            the 2nd sentence of every 14 sentences to condition 2;
#							...
#						 the 14th sentence of every 14 sentences to condition 14;
#

import sys, string

if len(sys.argv)!=4:
	print 'Usage: script [map][in][out]'
	sys.exit(1)

mapfile=sys.argv[1]
infile=sys.argv[2]
outfile=sys.argv[3]

tset=[]

fin=open(mapfile)
while True:
	sr=fin.readline()
	if sr=='':break
	tset.append((sr[sr.find(' ')+1:]).strip())
fin.close()

flst=[]
fin=open(infile)
while True:
	sr=fin.readline()
	if sr=='':break
	flst.append(sr.strip())
fin.close()

for ii in range(14):
	if tset[ii].endswith('_wv1'):
		for jj in range(ii, len(flst), 14):
			flst[jj]=string.replace(flst[jj], 'clean_wv1', tset[ii])
	else:
		for jj in range(ii, len(flst), 14):
			flst[jj]=string.replace(flst[jj], 'clean_wv1', tset[ii])
			flst[jj]=string.replace(flst[jj], '.wv1', '.wv2')

fout=open(outfile,'w')
for itm in flst:
	print >>fout, itm
fout.close()


