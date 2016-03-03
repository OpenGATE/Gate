/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateMergeManager_h
#define GateMergeManager_h 1
#include <iostream>
#include <string>
#include <vector>
#include <TFile.h>
#include <TChain.h>
#include <cstdlib>

using namespace std;

class GateMergeManager
{
public:

  GateMergeManager(bool fastMerge,int verboseLevel,bool forced,Long64_t maxRoot,string outDir){
     m_verboseLevel = verboseLevel;
     m_forced       =       forced;
     m_maxRoot      =      maxRoot;
     m_outDir       =       outDir;
     m_CompLevel    =            1;
     m_fastMerge    =    fastMerge;

     //check if a .Gate directory can be found
     if (!getenv("GC_DOT_GATE_DIR")) {
        cout<<"Environment variable GC_DOT_GATE_DIR not set !"<<endl;
        exit(1);
     }
     m_dir=getenv("GC_DOT_GATE_DIR");
     if (m_dir.substr(m_dir.length()-1,m_dir.length())=="/") m_dir=m_dir+".Gate/";
     else m_dir=m_dir+"/.Gate/";
     ifstream dirstream(m_dir.c_str());
     if (!dirstream) {
        cout<<"Failed to open .Gate directory"<<endl;
        exit(1);
     }
     dirstream.close();

  };
  ~GateMergeManager()
  {
   if (filearr) delete filearr;
  }


  void StartMerging(string splitfileName);
  void ReadSplitFile(string splitfileName);
  bool MergeTree(string name);
  bool MergeGate(TChain* chain);
  bool MergeSing(TChain* chain);
  bool MergeCoin(TChain* chain);

  // the cleanup after succesful merging
  void StartCleaning(string splitfileName,bool test);

  // the merging methods
  void MergeRoot();

private:
  void FastMergeRoot(); 
  bool FastMergeGate(string name);
  bool FastMergeSing(string name);
  bool FastMergeCoin(string name); 
  bool           m_forced;             // if to overwrite existing files
  int      m_verboseLevel;  
  TFile**         filearr;
  Long64_t      m_maxRoot;             // maximum size of root output file
  int         m_CompLevel;             // compression level for root output
  string            m_dir;             // .Gate directory path 
  string         m_outDir;             // where to save the output files
  int            m_Nfiles;             // number of files to mergw
  vector<int> m_lastEvents;            // latestevent from al files
  vector<string> m_vRootFileNames;     // names of root files to merge
  TFile*         m_RootTarget;         // root output file
  string         m_RootTargetName;     // name of target i.e. root output file
  bool           m_fastMerge;          // fast merge option, corrects the eventIDs locally
};


#endif
