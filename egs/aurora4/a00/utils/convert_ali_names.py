#!/usr/bin/env python
#
# convert the gzipped alignment file's index names
# used for generating clean pdf alignment of the corresponding
# multi-style data.
#

import sys, string, gzip 

if len(sys.argv) != 4:
	print "Usage: script [scp][oriali][resali]"
	sys.exit(1)

scpfile=sys.argv[1]
oriali=sys.argv[2]
resali=sys.argv[3]

names=[]
fin=open(scpfile)
while True:
	sr=fin.readline()
	if sr=='':break
	sr=(sr[:sr.find(' ')]).strip()
	names.append(sr)
fin.close()

fin=gzip.open(oriali, 'rb')
fout=gzip.open(resali, 'wb')
while True:
	sr=fin.readline()
	if sr=='':break
	sr=sr.strip()
	fname=(sr[:sr.find(' ')]).strip()
	cnt=(sr[sr.find(' ')+1:]).strip()
	if fname in names:
		print >>fout, fname+' '+cnt
	fname=string.replace(fname, '.wv1', '.wv2')
	if fname in names:
		print >>fout, fname+' '+cnt
fin.close()
fout.close()



