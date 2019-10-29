/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateSplitManager.hh"
#include <cstdlib>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

GateSplitManager::GateSplitManager(G4int nAliases,G4String* aliases,G4String platform,G4String pbsscript,
                                   G4String slurmscript,G4String condorscript,G4String macfile,G4int nSplits,G4int time)
{
 toPlatform = new GateToPlatform(nSplits,platform,pbsscript,slurmscript,condorscript,macfile,time);
 macParser  = new GateMacfileParser(macfile,nSplits,nAliases,aliases);
 numberOfSplits=nSplits;
}

void GateSplitManager::CheckEnvironment()
{
 G4String env="";
 G4int err=0;
 if (!getenv("GC_DOT_GATE_DIR")) err=1;
 if (!getenv("GC_GATE_EXE_DIR")) err=2;
 if (!getenv("PWD"))             err=3;

 if (err==1) cout<<"Environment variable GC_DOT_GATE_DIR not set !"<<endl; 
 if (err==2) cout<<"Environment variable GC_GATE_EXE_DIR not set !"<<endl;
 if (err==3) cout<<"Environment variable PWD not set !"<<endl;
 if (err>0) CleanAbort();
 
 env=getenv("GC_DOT_GATE_DIR");
 if (env.length()==0) err=4;
 env=getenv("GC_GATE_EXE_DIR");
 if (env.length()==0) err=5;
 env=getenv("PWD");
 if (env.length()==0) err=6;

 if (err==4) cout<<"Environment variable GC_DOT_GATE_DIR not set !"<<endl; 
 if (err==5) cout<<"Environment variable GC_GATE_EXE_DIR not set !"<<endl;
 if (err==6) cout<<"Environment variable PWD not set !"<<endl;
 if (err>0) CleanAbort(); 
}

void GateSplitManager::StartSplitting()
{
 CheckEnvironment();
 toPlatform->SetVerboseLevel(m_verboseLevel);
 macParser->SetVerboseLevel(m_verboseLevel);
 G4String outputMacDir;
 G4int err=0;
 G4String dir=getenv("GC_DOT_GATE_DIR");
 if (dir.substr(dir.length()-1,dir.length())=="/") dir=dir+".Gate/";
 else dir=dir+"/.Gate/";
 std::ifstream dirstream(dir.c_str());
 if (!dirstream) { 
 const G4String mkdir("mkdir "+dir); 
 if(m_verboseLevel>1)cout<<"Information : Creating a .Gate directory... "; 
 const int res = system(mkdir.c_str()); 
 if(m_verboseLevel>1)cout<<"done"<<endl;
 if (res!=0) 
  {
   cout<<"Error : Failed to create .Gate directory"<<endl;
   CleanAbort();
  }
 }

 dirstream.close();
 //call macParser to generate fully resolved macros + splitfile
 err=macParser->GenerateResolvedMacros(dir/*SIMON ,seeds*/);
 if (err) CleanAbort();
 else outputMacDir=macParser->GetOutputMacDir();
 //call toPlatform to generate submit file
 err=toPlatform->GenerateSubmitfile(outputMacDir);
 if (err) CleanAbort();
}

void GateSplitManager::CleanAbort()
{
 std::ofstream tmp1, tmp2;
 if (macParser) macParser->CleanAbort(tmp1,tmp2);
 cout<<"Made clean exit !"<<endl;  
 exit(1);
}

GateSplitManager::~GateSplitManager()
{
 delete macParser;
 delete toPlatform; 
}


