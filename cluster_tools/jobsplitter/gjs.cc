/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <queue>
#include <string>
#include <iostream>
#include <sstream>
#include "GateSplitManager.hh"
using std::string;
using namespace std;

void showhelp()
{
	cout<<endl;
	cout<<"  +-------------------------------------------+"<<endl;
	cout<<"  | gjs -- The GATE cluster job macro spliter |"<<endl;
	cout<<"  +-------------------------------------------+"<<endl;
	cout<<endl;
	cout<<"  Usage: gjs [-options] your_file.mac"<<endl;
	cout<<endl;
	cout<<"  Options (in any order):"<<endl;
	cout<<"  -a value alias             : use any alias"<<endl;
	cout<<"  -numberofsplits, -n   n    : the number of job splits; default=1"<<endl;
	cout<<"  -clusterplatform, -c  name : the cluster platform, name is one of the following:"<<endl;
	cout<<"                               openmosix - condor - openPBS - slurm - xgrid"<<endl;
	cout<<"                               This executable is compiled with "<<GC_DEFAULT_PLATFORM<<" as default"<<endl<<endl;
	cout<<"  -openPBSscript, os         : template for an openPBS script "<<endl;
	cout<<"                               see the example that comes with the source code (script/openPBS.script)"<<endl;
	cout<<"                               overrules the environment variable below"<<endl<<endl; 
	cout<<"  -slurmscript, ss           : template for a SLURM script "<<endl;
	cout<<"                               see the example that comes with the source code (script/slurm.script)"<<endl;
	cout<<"                               overrules the environment variable below"<<endl<<endl; 
	cout<<"  -condorscript, cs          : template for a condor submit file"<<endl;
	cout<<"                               see the example that comes with the source code (script/condor.script)"<<endl;
	cout<<"  -v                         : verbosity 0 1 2 3 - 1 default "<<endl;
	cout<<endl;
	cout<<"  Environment variables:"<<endl;
	cout<<"  GC_DOT_GATE_DIR  : indicates the .Gate directory for splitted mac files"<<endl;
	cout<<"  GC_GATE_EXE_DIR  : indicates the directory with the Gate executable"<<endl;
	cout<<"  GC_PBS_SCRIPT : the openPBS template script (!optionnal variable!)"<<endl;
	cout<<endl;
	cout<<"  Usage (bash):"<<endl;
	cout<<"    export GC_DOT_GATE_DIR=/home/user/gatedir/"<<endl;
	cout<<"    export GC_GATE_EXE_DIR=/home/user/gatedir/bin/Linux-g++/"<<endl;
	cout<<endl;
	cout<<"  Examples:"<<endl;
	cout<<"    gjs -numberofsplits 10 -clusterplatform openmosix macro.mac"<<endl;
	cout<<"    gjs -numberofsplits 10 -clusterplatform openmosix -a /somedir/rootfilename ROOT_FILE macro.mac"<<endl<<endl;
	cout<<"    gjs -numberofsplits 10 -clusterplatform openPBS -openPBSscript /somedir/script macro.mac"<<endl<<endl;
	cout<<"    gjs -numberofsplits 10 -clusterplatform xgrid macro.mac"<<endl<<endl;
	cout<<"    gjs -numberofsplits 10  /somedir/script macro.mac"<<endl<<endl;
	exit(0);
}

int main(int argc,char** argv)
{ 
	G4String* aliases=0;
	G4String platform="";
	G4String macfile;
	G4String pbsscript;
	G4String slurmscript;
	G4String condorscript;
	G4int nSplits=0;
	G4int nextArg = 1;
	G4int indicator=0;
	G4int nAliases=0;
	G4int time=0;
	G4int verb=1;
	aliases=new G4String[argc];
	
	int debug=0;
	if (argc==1) showhelp();
	// Parse the command line
	while (nextArg<argc)
	{
		indicator=0;
		int size=strlen(argv[nextArg]);
		if (!strcmp(argv[nextArg],"-a") && indicator==0)
		{
			if(debug)cout<<"found -a\n";
			indicator=2;
			aliases[nAliases]=argv[nextArg+1];
			nAliases++;
			aliases[nAliases]=argv[nextArg+2];
			nAliases++;
		}  
		if ((!strcmp(argv[nextArg],"-numberofsplits") || !strcmp(argv[nextArg],"-n")) && indicator==0)
		{
			indicator=1;
			stringstream ss(argv[nextArg+1]);
			ss>>nSplits;
			if(debug)cout<<"found -numberofsplits "<<nSplits<<endl;
		} 
		if (!strcmp(argv[nextArg],"-time") && indicator==0)
		{
			indicator=1;
			time=1;
			nextArg-=1;
			if(debug)cout<<"found -time "<<time<<endl;
		}   
		if (!strcmp(argv[nextArg],"-v") && indicator==0)
		{
			indicator=1;
			stringstream ss(argv[nextArg+1]);
			ss>>verb;
			if(debug)cout<<"found -v "<<verb<<endl;
		} 
		if ((!strcmp(argv[nextArg],"-clusterplatform") || !strcmp(argv[nextArg],"-c")) && indicator==0)
		{
			indicator=1;
			platform=argv[nextArg+1];
			if(debug)cout<<"found -clusterplatform "<<platform<<endl;
		}  
		if ((!strcmp(argv[nextArg],"-openPBSscript") || !strcmp(argv[nextArg],"-os")) && indicator==0)
		{
			indicator=1;
			pbsscript=argv[nextArg+1];
			if(debug)cout<<"found -openPBSscript "<<pbsscript<<endl;
		}  
		if ((!strcmp(argv[nextArg],"-slurmscript") || !strcmp(argv[nextArg],"-ss")) && indicator==0)
		{
			indicator=1;
			slurmscript=argv[nextArg+1];
			if(debug)cout<<"found -slurmscript "<<slurmscript<<endl;
		}  
		if ((!strcmp(argv[nextArg],"-condorscript") || !strcmp(argv[nextArg],"-cs")) && indicator==0)
		{
			indicator=1;
			condorscript=argv[nextArg+1];
			if(debug)cout<<"found -condorscript "<<condorscript<<endl;
		}  
		if (size>4 && indicator==0)
		{
			G4String ss(argv[nextArg]);
			if (ss.contains(".mac"))
			{
				indicator=1;
				macfile=argv[nextArg];
				nextArg-=1;
				if(debug)cout<<"found .mac "<<macfile<<endl;
			}
		}
		if (( !strcmp(argv[nextArg],"-help")||!strcmp(argv[nextArg],"-h") ) && indicator==0)
		{
			indicator=1;
			nextArg-=1;
			if(debug)cout<<"found -h(elp) "<<endl;
			showhelp();
			exit(0);
		}
		if (indicator==0)
		{
			// The argument was not recognised: exit
			cout<<"Argument: "<<argv[nextArg]<<" was not recognised as a valid option\n";
			exit(1);      
		}  
		else {
			if (indicator==2) nextArg+=3; 
			else nextArg+=2;  
		}
	} 
	
	if (platform=="" || platform=="openmosix" || platform=="openPBS" || platform=="slurm" || platform=="condor"|| platform=="xgrid")
	{  
		if (platform=="")
		{
			platform=GC_DEFAULT_PLATFORM;
			if(verb>1)cout<<"Information : using  "<<GC_DEFAULT_PLATFORM<<" as default cluster platform!"<<endl;
		}
		if (platform=="openPBS"&&pbsscript==""){
			if (getenv("GC_PBS_SCRIPT")){
				pbsscript=getenv("GC_PBS_SCRIPT");
				if(verb>1&&pbsscript!="")cout<<"Information : using $GC_PBS_SCRIPT="<<pbsscript<<" as default PBS template script"<<endl;
			}}  
		if (platform!="openPBS"&&pbsscript!="")
		{
			if(verb>0)cout<<"Warning : cluster platform is not openPBS, openPBSscript ignored!"<<endl;
		}
		if (platform=="openPBS"&&pbsscript=="")
		{
			cout<<"Error : cluster platform is openPBS but openPBSscript undefined!"<<endl;
			exit(1);
		}
		if ((platform=="openPBS"&&pbsscript!="")||platform=="openmosix"||(platform=="condor"&&condorscript!=""))
		{
			if(verb>1)cout<<"Information : using  "<<platform<<" as cluster platform!"<<endl;
		}
		if (platform!="condor"&&condorscript!="")
		{
			if(verb>0)cout<<"Warning : cluster platform is not condor, condorscript ignored!"<<endl;
		}
		if (platform=="condor"&&condorscript=="")
		{
			cout<<"Error : cluster platform is condor but condorscript is not supplied!"<<endl;
			exit(1);
		}
	}
	else {
		cout<<"Error : cluster platform not supported or invalid!"<<endl;
		exit(1);  
	}
	
	if(debug) cout<<"nSplits "<<nSplits<<endl;
	if (nSplits<=0)
	{
		nSplits=1;
		if(verb>0)cout<<"Warning : Invalid number of splits, using default=1!"<<endl; 
	}
	//create a splitmanager to coordinate it all  
	GateSplitManager* manager;
	manager=new GateSplitManager(nAliases,aliases,platform,pbsscript,slurmscript,condorscript,macfile,nSplits,time);
	manager->SetVerboseLevel(verb);
	manager->StartSplitting();
	
	delete[] aliases;   
	delete manager;
	return 0;
}


