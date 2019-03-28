/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


//#include <sys/types.h>
//#include <sys/stat.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <cstdlib> 
#include <sys/stat.h>

#include "GateToPlatform.hh"

using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::ostringstream;
              
GateToPlatform::GateToPlatform(G4int numberOfSplits, G4String thePlatform, G4String thePbsscript, G4String theSlurmScript, G4String theCondorScript, G4String outputMacName, G4int time)
{
	nSplits=numberOfSplits;
	platform=thePlatform;
	pbsScript=thePbsscript;
	slurmScript=theSlurmScript;
	condorScript=theCondorScript;
	useTiming=time;
	outputMacfilename=outputMacName.substr(0,outputMacName.length()-4);
}

GateToPlatform::~GateToPlatform()
{ 
}

int GateToPlatform::GenerateSubmitfile(G4String outputMacDir)
{
	G4int err=0;
	//check if a GC_GATE_EXE_DIR variable can be found
	G4String dir=getenv("GC_GATE_EXE_DIR");
	outputDir=outputMacDir;
	if (platform=="openmosix"){
		err=GenerateOpenMosixSubmitfile();
		if (err>0) return 1;
	} 
	if (platform=="openPBS"){
		err+=GenerateOpenPBSScriptfile();
		err+=GenerateOpenPBSSubmitfile();
		if (err>0) return 1;
	}
	if (platform=="slurm"){
		err+=GenerateSlurmScriptfile();
		err+=GenerateSlurmSubmitfile();
		if (err>0) return 1;
	}
	if (platform=="condor"){
		err+=GenerateCondorSubmitfile();
		if (err>0) return 1;
	}
	if (platform=="xgrid"){
		err+=GenerateXgridSubmitfile();
		if (err>0) return 1;
	} 
	return(0);
}



int GateToPlatform::GenerateOpenPBSScriptfile()
{
	G4String dir=getenv("GC_GATE_EXE_DIR");
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing directory
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return 1;
	}
	dirstream.close();
	
	//create script file to be submitted with qsub (PBS)
	
	
	char out_name[1000];
	G4String buf;
	for (G4int i=1;i<=nSplits;i++)
	{
		ostringstream cnt;
		cnt<<i;
		// open template script file
		ifstream inFile(pbsScript);
		if (!inFile) {
			cout<< "Error : could not access openPBS script template file! "<<pbsScript<< endl;
			return(1);
		}
		sprintf(out_name,"%s%i%s",outputDir.c_str(),i,".pbs");
		ofstream scriptFile(out_name);
		if (!scriptFile) {
			cout<< "Error : could not create script file! "<<out_name<< endl;
			return(1);
		}
		while(getline(inFile,buf)){
			if(buf.find("#")!=0||buf.find("#PBS")==0){
				unsigned int pos=buf.find("GC_WORKDIR");
				if(pos<buf.length())  buf=buf.substr(0,pos)+outputDir.substr(0,outputDir.rfind("/"))+buf.substr(pos+10);
				pos=buf.find("GC_LOG");
				if(pos<buf.length())  buf=buf.substr(0,pos)+"log"+cnt.str()+buf.substr(pos+6);
				pos=buf.find("GC_ERR");
				if(pos<buf.length())  buf=buf.substr(0,pos)+"err"+cnt.str()+buf.substr(pos+6);
				pos=buf.find("GC_JOBNAME");
				if(pos<buf.length())  {
					//openPBS max_jobname_length=15
					char jobname[16]="";
					strncpy(jobname,outputMacfilename.c_str(),15-cnt.str().length());
					buf=buf.substr(0,pos)+jobname+cnt.str()+buf.substr(pos+10);
				}
				pos=buf.find("GC_GATE");
				G4String timestr="";
				if (useTiming==1) timestr="time ";
				if(pos<buf.length())  buf=timestr+dir+"Gate "+outputDir+cnt.str()+".mac"+buf.substr(pos+7);
			}
			scriptFile<<buf<<endl;
		}
		scriptFile.close();
		inFile.close();
	}
	return 0; 
}

int GateToPlatform::GenerateOpenPBSSubmitfile()
{
	G4String dir=getenv("GC_GATE_EXE_DIR");
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing director
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return 1;
	}
	dirstream.close();
	
	G4String submitFilename=outputMacfilename+".submit";
	ofstream submitFile(submitFilename.c_str());
	if (!submitFile) {
		cout<< "Error : could not create submit file! "<<submitFilename<< endl;
		return(1);
	}
	submitFile<<"#! /bin/sh"<<endl;
	for (G4int i=1;i<=nSplits;i++)
	{
		submitFile<<"echo qsub "<<outputDir<<i<<+".pbs"<<endl;
		submitFile<<"qsub "<<outputDir<<i<<+".pbs"<<endl;
	}
	submitFile.close();
	chmod( submitFilename.c_str() , S_IRWXU|S_IRGRP );
	return 0;
}

int GateToPlatform::GenerateSlurmScriptfile()
{
	G4String dir=getenv("GC_GATE_EXE_DIR");
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing directory
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return 1;
	}
	dirstream.close();
	
	//create script file to be submitted with sbatch (SLURM)
	
	
	char out_name[1000];
	G4String buf;
	for (G4int i=1;i<=nSplits;i++)
	{
		ostringstream cnt;
		cnt<<i;
		// open template script file
		ifstream inFile(slurmScript);
		if (!inFile) {
			cout<< "Error : could not access SLRUM script template file! "<<slurmScript<< endl;
			return(1);
		}
		sprintf(out_name,"%s%i%s",outputDir.c_str(),i,".slm");
		ofstream scriptFile(out_name);
		if (!scriptFile) {
			cout<< "Error : could not create script file! "<<out_name<< endl;
			return(1);
		}
		while(getline(inFile,buf)){
			if(buf.find("#")!=0||buf.find("#SBATCH")==0){
				unsigned int pos=buf.find("GC_WORKDIR");
				if(pos<buf.length())  buf=buf.substr(0,pos)+outputDir.substr(0,outputDir.rfind("/"))+buf.substr(pos+10);
				pos=buf.find("GC_LOG");
				if(pos<buf.length())  buf=buf.substr(0,pos)+"log"+cnt.str()+buf.substr(pos+6);
				pos=buf.find("GC_ERR");
				if(pos<buf.length())  buf=buf.substr(0,pos)+"err"+cnt.str()+buf.substr(pos+6);
				pos=buf.find("GC_JOBNAME");
				if(pos<buf.length())  {
					//SLURM max_jobname_length <= 1024 chars
					char jobname[256]="";
					strncpy(jobname,outputMacfilename.c_str(),255-cnt.str().length());
					buf=buf.substr(0,pos)+jobname+cnt.str()+buf.substr(pos+10);
				}
				pos=buf.find("GC_GATE");
				G4String timestr="";
				if (useTiming==1) timestr="time ";
				if(pos<buf.length())  buf=timestr+dir+"Gate "+outputDir+cnt.str()+".mac"+buf.substr(pos+7);
			}
			scriptFile<<buf<<endl;
		}
		scriptFile.close();
		inFile.close();
	}
	return 0; 
}

int GateToPlatform::GenerateSlurmSubmitfile()
{
	G4String dir=getenv("GC_GATE_EXE_DIR");
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing director
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return 1;
	}
	dirstream.close();
	
	G4String submitFilename=outputMacfilename+".submit";
	ofstream submitFile(submitFilename.c_str());
	if (!submitFile) {
		cout<< "Error : could not create submit file! "<<submitFilename<< endl;
		return(1);
	}
	submitFile<<"#! /bin/sh"<<endl;
	for (G4int i=1;i<=nSplits;i++)
	{
		submitFile<<"echo sbatch "<<outputDir<<i<<+".slm"<<endl;
		submitFile<<"sbatch "<<outputDir<<i<<+".slm"<<endl;
	}
	submitFile.close();
	chmod( submitFilename.c_str() , S_IRWXU|S_IRGRP );
	return 0;
}

int GateToPlatform::GenerateOpenMosixSubmitfile()
{
	G4String dir=getenv("GC_GATE_EXE_DIR");
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing directory
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return(1);
	}
	dirstream.close();
	
	G4String submitFilename=outputMacfilename+".submit";
	ofstream submitFile(submitFilename.c_str());
	if (!submitFile) {
		cout<< "Error : could not create submit file! "<<submitFilename<< endl;
		return(1);
	}
	submitFile<<"#! /bin/sh"<<endl;
	for (G4int i=1;i<=nSplits;i++)
	{
		if (useTiming==1) submitFile<<"\\time "<<dir+"Gate "<<outputDir<<i<<+".mac"<<" 2>timefile"<<i<<" &"<<endl; 
		else submitFile<<dir+"Gate "<<outputDir<<i<<+".mac"<<" &"<<endl;
		submitFile<<"sleep 10s"<<endl;  
	}
	submitFile.close();
	return 0;
}

int GateToPlatform::GenerateCondorSubmitfile()
{
	char buffer[256];
	G4String scriptline;
	G4String dir=getenv("GC_GATE_EXE_DIR");
	G4int noCopy=0; 
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing directory
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return(1);
	}
	dirstream.close();
	
	G4String submitFilename=outputMacfilename+".submit";
	ofstream submitFile(submitFilename.c_str());
	if (!submitFile) {
		cout<< "Error : could not create submit file! "<<submitFilename<< endl;
		return(1);
	}
	ifstream scriptFile(condorScript.c_str());
	if (!scriptFile) {
		cout<< "Error : could not open the condor script file! "<<condorScript<< endl;
		return(1);
	}
	while(!scriptFile.eof())
	{
		scriptFile.getline(buffer,256);
		scriptline=buffer;
		if (scriptline.contains("#GJS PART => DO NOT REMOVE")!=0) noCopy=1;
		if (noCopy==0) 
		{
			if (scriptline.contains("Executable")!=0 && scriptline.contains("$GC_EXEC")!=0) 
				submitFile<<"Executable     = "<<dir+"Gate"<<endl;
			else submitFile<<scriptline<<endl;
			
		}
	}
	submitFile<<endl;
	for (G4int i=1;i<=nSplits;i++)
	{
		submitFile<<"Arguments      = "<<outputDir<<i<<+".mac"<<endl;                                              
		submitFile<<"Output         = "<<outputDir<<i<<+".out"<<endl;
		submitFile<<"Error          = "<<outputDir<<i<<+".err"<<endl;
		submitFile<<"Log            = "<<outputDir<<i<<+".log"<<endl;
		submitFile<<"Queue"<<endl<<endl;   
	}
	scriptFile.close();
	submitFile.close();
	return 0;
}
int GateToPlatform::GenerateXgridSubmitfile()
{
	G4String dir=getenv("GC_GATE_EXE_DIR");
	if (dir.substr(dir.length()-1,dir.length())!="/") dir=dir+"/"; 
	
	//check if we have an existing directory
	ifstream dirstream(dir.c_str());
	if (!dirstream) { 
		cout<<"Error : Failed to detect the Gate executable directory"<<endl;
		cout<<"Please check your environment variables!"<<endl; 
		cout<<"Generated submit file may be invalid..."<<endl;  
		return(1);
	}
	dirstream.close();
		G4String submitXgridFilename=outputMacfilename+".plist";
	ofstream submitXgridFile(submitXgridFilename.c_str());
	if (!submitXgridFile) {
		cout<< "Error : could not create submit file! "<<submitXgridFilename<< endl;
		return(1);
	}
	submitXgridFile << "{"<<endl;
	submitXgridFile << "jobSpecification = {"<<endl;
	submitXgridFile << "applicationIdentifier = \"com.apple.xgrid.cli\";"<<endl;
	submitXgridFile << "inputFiles = {};"<< endl;
	submitXgridFile << "name = \"Gate\";"<< endl;
	submitXgridFile << "submissionIdentifier = abc;"<< endl;
	submitXgridFile << "taskSpecifications = {"<<endl;
	for (int i=0;i<nSplits;i++){
		submitXgridFile << i<< +" = {arguments =("<<outputDir<<i<<+".mac); command =\""<< dir<< +"Gate\"; }; "<< endl;
		}
	submitXgridFile << "};"<<endl;
	submitXgridFile << "};"<<endl;
	submitXgridFile << "}"<<endl;
	submitXgridFile.close();
	return 0;
}


