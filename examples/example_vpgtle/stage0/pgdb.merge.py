#!/usr/bin/env python

'''
This script using pyROOT and clitkMergeRootFiles (must be in your path) to merge any number of db-*.root files produced with gate_run_submit_cluster.sh under the specified run-dir (recursively) into pgdb.root.
'''

import os,sys,ROOT as r

if len(sys.argv) < 3:
	print "Specify 1) a number of jobs per material and 2) a run-dir with rootfiles with phasespaces."
	sys.exit()

cmdforclitk = ""
#rootfiles = [x.strip() for x in os.popen("find "+sys.argv[-1]+" -print | grep -i 'db-.*.root$'").readlines()]
rootfiles = [x.strip() for x in os.popen("find "+sys.argv[-1]+" -print | grep -i 'run.*/output.*/db.*.root$'").readlines()]
for rootfile in rootfiles:
	cmdforclitk=cmdforclitk+" -i "+rootfile.strip()
print "Found",len(rootfiles),"rootfiles, starting merge..."
os.popen("rm pgdb.root")
print "clitkMergeRootFiles"+cmdforclitk+" -o pgdb.root"
os.popen("clitkMergeRootFiles"+cmdforclitk+" -o pgdb.root")

##now, correct for number of jobs/matieral
if sys.argv[-2] is not "1":
	factor = 1./float(sys.argv[-2])
	tfile=r.TFile("pgdb.root","UPDATE")
	for item in tfile.GetListOfKeys():
		for thist in item.ReadObj().GetListOfKeys():
			name = thist.GetName()
			if 'EpEpgNorm' in name or 'GammaZ' in name:
				print name
				thist.ReadObj().Scale(factor)
	tfile.Write()
	print "PGDB scaled by factor",factor

print "Done! Your database can be found in db.root."
