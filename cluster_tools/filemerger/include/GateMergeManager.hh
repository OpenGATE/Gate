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
#include <cstdlib>
#include <TFile.h>
#include <TChain.h>

class GateMergeManager
{
public:

  GateMergeManager(bool fastMerge,int verboseLevel,bool forced,Long64_t maxRoot,std::string outDir){
     m_verboseLevel = verboseLevel;
     m_forced       =       forced;
     m_maxRoot      =      maxRoot;
     m_outDir       =       outDir;
     m_CompLevel    =            1;
     m_fastMerge    =    fastMerge;

     //check if a .Gate directory can be found
     if (!getenv("GC_DOT_GATE_DIR")) {
        std::cout<<"Environment variable GC_DOT_GATE_DIR not set !"<<std::endl;
        exit(1);
     }
     m_dir=getenv("GC_DOT_GATE_DIR");
     if (m_dir.substr(m_dir.length()-1,m_dir.length())=="/") m_dir=m_dir+".Gate/";
     else m_dir=m_dir+"/.Gate/";
     std::ifstream dirstream(m_dir.c_str());
     if (!dirstream) {
        std::cout<<"Failed to open .Gate directory"<<std::endl;
        exit(1);
     }
     dirstream.close();

  };
  ~GateMergeManager()
  {
   if (filearr) delete filearr;
  }


  void StartMerging(std::string splitfileName);
  void ReadSplitFile(std::string splitfileName);
  bool MergeTree(std::string name);
  bool MergeGate(TChain* chain);
  bool MergeSing(TChain* chain);
  bool MergeCoin(TChain* chain);

  // the cleanup after succesful merging
  void StartCleaning(std::string splitfileName,bool test);

  // the merging methods
  void MergeRoot();

private:
  void FastMergeRoot(); 
  bool FastMergeGate(std::string name);
  bool FastMergeSing(std::string name);
  bool FastMergeCoin(std::string name); 
  bool                 m_forced;             // if to overwrite existing files
  int            m_verboseLevel;  
  TFile**               filearr;
  Long64_t            m_maxRoot;             // maximum size of root output file
  int               m_CompLevel;             // compression level for root output
  std::string             m_dir;             // .Gate directory path 
  std::string          m_outDir;             // where to save the output files
  int                  m_Nfiles;             // number of files to mergw
  std::vector<int> m_lastEvents;             // latestevent from al files
  std::vector<std::string> m_vRootFileNames; // names of root files to merge
  TFile*           m_RootTarget;             // root output file
  std::string  m_RootTargetName;             // name of target i.e. root output file
  bool              m_fastMerge;             // fast merge option, corrects the eventIDs locally
};


#endif
