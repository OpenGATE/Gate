/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <TROOT.h>
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TBranch.h>
#include <TIterator.h>
#include <TObject.h>
#include <TKey.h>
#include <TH1.h>
#include <TH2.h>

#include <iostream>
#include <fstream>
#include <string>
#include <glob.h>
#include <ctype.h>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "GateMergeManager.hh"

using namespace std;

void GateMergeManager::StartMerging(string splitfileName){

  // get the files to merge
  ReadSplitFile(splitfileName);
  //do the merging
  if (m_fastMerge==true) FastMergeRoot();
  else MergeRoot();

  //if we are here the merging has been successful
  //we mark the directory as ready for cleanup
  //string dir   = splitfileName.substr(0,splitfileName.rfind("/"));
  //string ready = "touch "+dir+"/ready_for_delete";
  //const int res=system(ready.c_str());
  //if(res) cout<<"Strange?? Can't mark "<<dir<<" as done!"<<endl;
}

// to process the splitfile
void GateMergeManager::ReadSplitFile(string splitfileName){

  ifstream splitfile;
  splitfile.open(splitfileName.c_str());
  if(!splitfile){
     cout<<"Can't open split file: "<<splitfileName<<endl;
     exit(0);
  };

  //avoid multiple calls
  if(m_vRootFileNames.size()>0) return;

  stringstream ssnfiles;
  char cline[512];
  while(splitfile){               //we allow for rubbish before "Number of files:"
     splitfile.getline(cline,512);
     if(!strncmp(cline,"Number of files:",16)){
         ssnfiles.str(cline+16);
         ssnfiles>>m_Nfiles;      // the number of files to merge
         break;
     }
  }

  // now we look for the file names
  int iRoot=0;
  while(splitfile){
     splitfile.getline(cline,512);
     // check for root
     // input files
     if(!strncmp(cline,"Root filename:",14)){
        m_vRootFileNames.push_back(strtok(cline+15," "));
        m_vRootFileNames[iRoot]+=".root";
        if(m_verboseLevel>2) cout<<"Root input file name: "<<m_vRootFileNames[iRoot]<<endl;
        iRoot++;
     }
     // output file
     else if(!strncmp(cline,"Original Root filename:",23)){
        m_RootTargetName=strtok(cline+23," ");
        m_RootTargetName+=".root";
        if(m_outDir!=""){// output directory option used
           //replace path with outDir
           
           size_t pos=m_RootTargetName.rfind('/',m_RootTargetName.length());
           if(pos==string::npos) pos=-1;
           m_RootTargetName=m_outDir+m_RootTargetName.substr(pos+1);
        }
        if(m_verboseLevel>2) cout<<"Root output file name: "<<m_RootTargetName<<endl;
     }
  }

  // check if number of root files correct
  if(iRoot && iRoot!=m_Nfiles) {
       cout<<"Inconsistent number of root file entries in split file!"<<endl;
       exit(0);
  }
}

/************************************************************************************/
void GateMergeManager::FastMergeRoot()
{
   //check target name
   if(m_RootTargetName=="") {
      cout<<"No root output file name given in split file"<<endl;
      exit(0);
   }

   // number of files to merge
   int nfiles=m_vRootFileNames.size();
   if(nfiles<1){
      cout<<"No files to merge!"<<endl;
      exit(0);
   }
   //we try to recover the last_event_ID in all root files
   filearr=new TFile*[nfiles];
   for(int i=0;i<nfiles;i++) {
      filearr[i] = TFile::Open(m_vRootFileNames[i].c_str(),"OLD");
      if(filearr[i]==NULL){
         cout<<"Not a readable file "<<m_vRootFileNames[i]<<" - exit!"<<endl;   
         exit(0);
      }
   }  
   m_lastEvents.resize(nfiles+1);
   m_lastEvents[0]=0;
   bool flgLstEvntID(false);

   // latest_event_ID histogram
   flgLstEvntID=true;
   TH1* h=NULL;
   for(int i=0;i<nfiles;i++){
     h = (TH1*)filearr[i]->Get("latest_event_ID");
     m_lastEvents[i+1]=int(h->GetMean());
     }

   if(!flgLstEvntID) {
      cout<<"FastMerge: No latest_event_ID histogram - exit!"<<endl;
      exit(0);
   }

 /*for (int i=0;i<nfiles;i++)
  {
   cout<<"Last ID is: "<<m_lastEvents[i+1]<<" File " <<i+1<<endl;
  }*/

   //find all trees
   vector<string> treeNames;
   TFile* node = TFile::Open(m_vRootFileNames[0].c_str(),"OLD");
   TIter nextkey(node->GetListOfKeys());
   TKey *key=0;
   while ((key = (TKey*)nextkey()))
   {
     node->cd();
     TObject* obj = (TObject*)key->ReadObj();
     if(  obj->IsA()->InheritsFrom("TTree"))
     {
       //no doubles
       bool singleName=true;
       for(unsigned int i=0;i<treeNames.size();i++)
       {
         if(treeNames[i].compare(obj->GetName())==0)
         {
           singleName=false;
           break;
         }
       }
       if(singleName) treeNames.push_back(obj->GetName());
     }
   }
   //take care of all trees
   for(unsigned int i=0;i<treeNames.size();i++) 
   {
     if(!MergeTree(treeNames[i])) if(m_verboseLevel>1) cout<<"Problem with merging "<<treeNames[i]<<endl; 
   }
}
/************************************************************************************/
void GateMergeManager::MergeRoot(){

   //check target name
   if(m_RootTargetName=="") {
      cout<<"No root output file name given in split file"<<endl;
      exit(0);
   }

   // number of files to merge
   int nfiles=m_vRootFileNames.size();

   // the root output file
   if(m_forced) {
      m_RootTarget = TFile::Open(m_RootTargetName.c_str(),"RECREATE");
   } else {
      m_RootTarget = TFile::Open(m_RootTargetName.c_str(),"NEW");
      if(!m_RootTarget){
          cout<<"The root ouput file already exists! Try -f to overwrite and erase the file."<<endl;
          exit(0);
      }
   }

   if(nfiles<1){
      cout<<"No files to merge!"<<endl;
      exit(0);
   }
   
   filearr=new TFile*[nfiles];
   //TFile* filearr[nfiles];
   for(int i=1;i<nfiles;i++) {
      filearr[i] = TFile::Open(m_vRootFileNames[i].c_str(),"OLD");
      if(filearr[i]==NULL){
         cout<<"Not a readable file "<<m_vRootFileNames[i]<<" - exit!"<<endl;   
         exit(0);
      }
   }

// first we copy all histos (only top directory) 
// and look for the latestEventID
// and find out which  trees/ntuples  exist
   m_lastEvents.resize(nfiles+1);
       m_lastEvents[0]=0;

   bool flgLstEvntID(false);

   if(m_verboseLevel>0) {
      cout<<"Combining: ";
      for(int i=0;i<nfiles;i++) cout<<m_vRootFileNames[i]<<" ";
      cout<<"-> "<<m_RootTarget->GetName()<<endl;
   }

   vector<string> treeNames;
   TFile* node = TFile::Open(m_vRootFileNames[0].c_str(),"OLD");
   if(node==NULL){
       cout<<"Not a readable file "<<m_vRootFileNames[0]<<" - exit!"<<endl;   
       exit(0);
   }
        TIter nextkey(node->GetListOfKeys());
        TKey *key=0;
        while ((key = (TKey*)nextkey())) {
          node->cd();
          TObject* obj = (TObject*)key->ReadObj();
          if(  obj->IsA()->InheritsFrom("TH1")){
             m_RootTarget->cd();
             TH1 *h1 = (TH1 *)obj->Clone();
             if(strcmp(h1->GetName(),"latest_event_ID")){
                     //any other histogram
                     for(int i=1;i<nfiles;i++){
                          TH1 *h2 = (TH1*)filearr[i]->Get( h1->GetName() );
                          h1->Add( h2 ); 
                     }
                     h1->Write();
             } else {
                          // latest_event_ID histogram
                          flgLstEvntID=true;
                          m_lastEvents[1]=int(h1->GetMean());
                          TH1* h=NULL;
                          for(int i=1;i<nfiles;i++){
                                  h = (TH1*)filearr[i]->Get("latest_event_ID");
                                  m_lastEvents[i+1]=int(h->GetMean());
                          }
                          if(h!=NULL) h->Write(); // we keep the highest latest_event_ID to allow for successive merging
                          else h1->Write();
              }
         }
         if(  obj->IsA()->InheritsFrom("TH2")){
           m_RootTarget->cd();
           TH2 *h1 = (TH2 *)obj->Clone();
                        for(int i=1;i<nfiles;i++){
                                TH2 *h2 = (TH2*)filearr[i]->Get( h1->GetName() );
                                h1->Add( h2 ); 
                        }
                        h1->Write();
                }
         if(  obj->IsA()->InheritsFrom("TTree")){
           //no doubles
           bool singleName=true;
           for(unsigned int i=0;i<treeNames.size();i++) {
              if(treeNames[i].compare(obj->GetName())==0){
                 singleName=false;
                 break;
              }
           }
           if(singleName) treeNames.push_back(obj->GetName());
         }
   }
   if(!flgLstEvntID) {
      cout<<"Merge: No latest_event_ID histogram - exit!"<<endl;
      exit(0);
   }

   //now we take care of the trees
   for(unsigned int i=0;i<treeNames.size();i++) 
     if(!MergeTree(treeNames[i])) if(m_verboseLevel>1) cout<<"Problem with merging "<<treeNames[i]<<endl; 

   // everything is done
}

/*******************************************************************************************/
// cleanup after merging
void GateMergeManager::StartCleaning(string splitfileName,bool test){

  // get the files to erase
  ReadSplitFile(splitfileName);

  string dir=splitfileName.substr(0,splitfileName.rfind("/"));

  // test for the mark: ready_for_delete 
  string touched=dir+"/ready_for_delete";
  ifstream ready;
  ready.open(touched.c_str());
  if (!ready) {
     cout<<"Cannot do the cleanup - directory "<<dir<<" not marked as ready!"<<endl;
     if(!test) exit(0);
  }

  // print info
  if(test)                    cout<<"I would like to erase:"<<endl;
  if(!test&&m_verboseLevel>1) cout<<"Going to erase the following files:"<<endl;
  if(m_verboseLevel>1||test){
     cout<<" --> "<<dir<<endl;
     // root
     for(unsigned int i=0;i<m_vRootFileNames.size();i++){
        cout<<" --> "<<m_vRootFileNames[i]<<endl;
     }
  }

  // erase!
  if(!test){
     for(unsigned int i=0;i<m_vRootFileNames.size();i++){
        const string rmfiles("rm -f "+m_vRootFileNames[i]);
        const int res = system(rmfiles.c_str());
        if(res) {
               cout<<"Could not remove "<<m_vRootFileNames[i]<<endl;
               cout<<"Please remove it manually!"<<endl;
        }
     }
     const string rmdir="rm -rf "+dir;
     if(system(rmdir.c_str())) {
            cout<<"Could not remove "<<dir<<endl;
            cout<<"Please remove it manually!"<<endl;
     }
  }
}

/*******************************************************************************************/
//find out what kind of tree we have and call the right merger
bool GateMergeManager::MergeTree(string chainName){
if (m_fastMerge==false)
 {
   TChain* chain = new TChain(chainName.c_str());

   // number of files to merge
   int nfiles=m_vRootFileNames.size();

   for(int i=0;i<nfiles;i++) chain->Add(m_vRootFileNames[i].c_str());
   int nentries=chain->GetEntries();
   if(nentries<=0) {
      if(m_verboseLevel>1) cout<<chain->GetName()<<" is empty!"<<endl;
      return false;
   }

   if(chainName=="Gate") MergeGate(chain);

   if(chain->FindBranch("eventID1")!=NULL) {
     if(  (chain->FindBranch("runID")==NULL)
        ||(chain->FindBranch("time1")==NULL) 
        ||(chain->FindBranch("time2")==NULL) ) {
        cout<<"Cannot find one of:  runID, time1, time2 in "<<chain->GetName()<<endl;
        return false;
     }
     // coincidence
     return MergeCoin(chain);

   } else if(  (chain->FindBranch("runID")==NULL)||(chain->FindBranch("eventID")==NULL)
                                                  ||(chain->FindBranch("time")==NULL) )
            {
             cout<<"Cannot find one of: runID, eventID, time in "<<chain->GetName()<<endl;
             return false;
            }
           // Singles or Hits
           return MergeSing(chain);
   }
   else
   {
     if (chainName.find("Hits",0)!=string::npos) return FastMergeSing(chainName);
     if (chainName.find("Singles",0)!=string::npos) return FastMergeSing(chainName);
     if (chainName.find("Gate",0)!=string::npos) return FastMergeGate(chainName);
     if (chainName.find("Coincidences",0)!=string::npos) return FastMergeCoin(chainName);
   }
   return 0;
}

/*******************************************************************************************/
bool GateMergeManager::FastMergeGate(string name)
{
//create the output file
float   event   = 0;
int offset      = 0;
int currentfile = 0;
float lastEvent =-1;
//float maxtime   =-999999999;
string clusterName=name+"_cluster";

for(int j=0;j<m_Nfiles;j++)
 {
   filearr[j]->ReOpen("UPDATE");
   TTree *oldTree = (TTree*)gDirectory->Get(name.c_str());
   TBranch *branch  = oldTree->GetBranch("event");
   branch->SetAddress(&event);
   TBranch* newBranch=oldTree->Branch("eventcluster",&event,"event");

   Int_t nentries=(Int_t)oldTree->GetEntries();

   if (j==0) offset=0;
   else offset+=m_lastEvents[currentfile];
   currentfile++;

   for(int i=0;i<nentries;i++)
      {
       branch->GetEvent(i);
       if(lastEvent!=event)
        {
         lastEvent=event;
         event+=offset;
         newBranch->Fill();
        } 
       else
        {
         if((i==nentries-1) && (j==m_Nfiles-1))
          {
           event+=offset;
           newBranch->Fill();
          }
         else if (i!=nentries-1) 
              { 
               branch->GetEvent(i);
               event+=offset;
               newBranch->Fill();
               lastEvent=0;
               offset=0;
              }
        }
      }
    oldTree->Write();
 }
 return true;
}

/*******************************************************************************************/
bool GateMergeManager::FastMergeSing(string name)
{
int eventID = 0;
int runID = 0; 
int offset      = 0;
int currentfile = 0;
float lastRun =-1;
string clusterName=name+"_cluster"; 

for(int j=0;j<m_Nfiles;j++)
 {
   filearr[j]->ReOpen("UPDATE");
   TTree *oldTree = (TTree*)gDirectory->Get(name.c_str());
   TBranch *branch  = oldTree->GetBranch("eventID");
   branch->SetAddress(&eventID);
   TBranch *branch2  = oldTree->GetBranch("runID");
   branch2->SetAddress(&runID);
   TBranch* newBranch=oldTree->Branch("eventIDcluster",&eventID,"eventID/I");

   Int_t nentries=(Int_t)oldTree->GetEntries();

   if (j==0) offset=0;
   else offset+=m_lastEvents[currentfile];
   currentfile++;

   for(int i=0;i<nentries;i++)
     {
      branch->GetEvent(i);
      if(lastRun!=runID)
        {
         lastRun=runID;
         offset=0;
        }
      eventID+=offset;
      newBranch->Fill();
    }
  oldTree->Write();
  }
  return true;
}

/*******************************************************************************************/
bool GateMergeManager::FastMergeCoin(string name)
{
//create the output file
int eventID1 = 0;
int eventID2 = 0;
int runID = 0;
int offset      = 0;
int currentfile = 0;
float lastRun =-1;
string clusterName=name+"_cluster";
cout<<"starting coincidence merging..."<<endl;
for(int j=0;j<m_Nfiles;j++)
 {
cout<<"working on file..."<<j<<endl;
   filearr[j]->ReOpen("UPDATE");
   TTree *oldTree = (TTree*)gDirectory->Get(name.c_str());
   TBranch *branch1  = oldTree->GetBranch("eventID1");
   branch1->SetAddress(&eventID1);
   TBranch *branch2  = oldTree->GetBranch("eventID2");
   branch2->SetAddress(&eventID2);
   TBranch *branch3  = oldTree->GetBranch("runID");
   branch3->SetAddress(&runID);

   TBranch* newBranch1=oldTree->Branch("eventID1cluster",&eventID1,"eventID1/I");
   TBranch* newBranch2=oldTree->Branch("eventID2cluster",&eventID2,"eventID2/I");

   Int_t nentries=(Int_t)oldTree->GetEntries();

   if (j==0) offset=0;
   else offset+=m_lastEvents[currentfile];
   cout<<"the offset in this file is: "<<offset<<endl;
   currentfile++;

   for(int i=0;i<nentries;i++){
       branch1->GetEvent(i);
       branch2->GetEvent(i);
       branch3->GetEvent(i);
       if(lastRun!=runID) 
        {
         lastRun=runID;
         offset=0; 
        }
       eventID1+=offset;
       eventID2+=offset;
       newBranch1->Fill();
       newBranch2->Fill();
    }
    oldTree->Write();
 }
    return true;
}

/*******************************************************************************************/
// Gate tree merger
bool GateMergeManager::MergeGate(TChain* chainG) {

   int nentries=chainG->GetEntries();   

   float   event   = 0;
   chainG->SetBranchAddress("event",&event);
   Float_t iontime = 0;
   chainG->SetBranchAddress("iontime",&iontime);

   //create the new tree
   //Allow for large files and do not write every bit separately
   m_RootTarget->cd();
   TTree * newTree = chainG->CloneTree(0);
   newTree->SetAutoSave(2000000000);
   if(m_maxRoot!=0) newTree->SetMaxTreeSize(m_maxRoot);
   else newTree->SetMaxTreeSize(17179869184LL);

   // changing the compression level everywhere
   TBranch *br;
   TIter next(newTree->GetListOfBranches());
   while ((br=(TBranch*)next())) br->SetCompressionLevel(m_CompLevel);

   // the main part: modify the eventID
   int offset      = 0;
   int currentTree = 0;
   float lastEvent =-1;
   float maxtime   =-999999999;
   for(int i=0;i<nentries;i++){
       if (chainG->GetEntry(i) <= 0) break; //end of chain
       if(chainG->GetTreeNumber()>currentTree) {
           currentTree++;
           offset+=m_lastEvents[currentTree];
           // check for overlaping time intervalls between different files
           // (not within the same file i.e. no time order assumed)
           if(iontime<maxtime)
              if(m_verboseLevel>0) 
                 cout<<"Warning - overlapping Gate iontime ("
                     <<iontime<<") in file: "<<m_vRootFileNames[currentTree].c_str()<<endl;
       }
       // run end is marked by repeating event
       if(lastEvent!=event) {
           lastEvent=event;
//           if (fpclassify(iontime)!=FP_NAN && fpclassify(iontime)!=FP_INFINITE) {
           if(finite(iontime)){
             if(maxtime<iontime) maxtime=iontime;
             // the offset to get a unique event numbering
             event+=offset;
             newTree->Fill();
           } else {
             cout<<"Warning - inf or NaN in Gate tree for iontime! "<<m_vRootFileNames[currentTree].c_str()<<endl;
             //we are lost anyhow
             maxtime   =-999999999;
           }
        } else {
           // run end 
           if(chainG->GetEntry(i+1) <= 0)  {              //chain end fill the double event 
              event+=offset;
              newTree->Fill();
           } else if(chainG->GetTreeNumber()==currentTree             //no new file or
                    ||(chainG->GetTreeNumber()>currentTree
                       &&m_lastEvents[chainG->GetTreeNumber()]==0) ) { //new file with no offset fill double event
              chainG->GetEntry(i);
              event+=offset;
              newTree->Fill();
              lastEvent=0;// we may have triple 1 (1 event run)
              offset=0;
           }
        }
      }
      newTree->Write();
      delete newTree;
      return true;
}

/*******************************************************************************************/
// Singles and Hits tree merger
bool GateMergeManager::MergeSing(TChain* chainS){

   int nentriesS=chainS->GetEntries();

   int eventID = 0;
   int runID   = 0;
   double time = 0;

   chainS->SetBranchAddress("eventID",&eventID);
   chainS->SetBranchAddress("runID",&runID);
   chainS->SetBranchAddress("time",&time);

   m_RootTarget->cd();
   TTree * newSing = chainS->CloneTree(0);
   newSing->SetAutoSave(2000000000);
   if(m_maxRoot!=0) newSing->SetMaxTreeSize(m_maxRoot);
   else newSing->SetMaxTreeSize(17179869184LL);

   // changing CompLevel everywhere
   TBranch *br;
   TIter next(newSing->GetListOfBranches());
   while ((br=(TBranch*)next())) br->SetCompressionLevel(m_CompLevel);

   //the main part: changing eventID
   int offset      = 0;
   int currentTree = 0;
   float lastRun   = 0;
   float maxtime   =-999999999;
   for(int i=0;i<nentriesS;i++){
         if(chainS->GetEntry(i) <= 0) break; //end of chain
         if(chainS->GetTreeNumber()>currentTree) {
           currentTree++;
            offset+=m_lastEvents[currentTree];
           // check for overlaping time intervalls between different files
           // (not within the same file i.e. no time order assumed)
           if(time<maxtime) 
               if(m_verboseLevel>0) 
                  cout<<"Warning - overlapping Singles time ("
                      <<time<<") in file: "<<m_vRootFileNames[currentTree].c_str()<<endl;
         }
         // the offset to get a unique event numbering
         if(lastRun!=runID) {
            // run end 
            lastRun=runID;
            offset=0;     //new run in file we must not change the eventID anymore
         }
         eventID+=offset;
         newSing->Fill();
         if(maxtime<time)maxtime=time;
    }
    newSing->Write();
    delete newSing;
    return true;
}

/*******************************************************************************************/
// Coincidences tree merger
bool GateMergeManager::MergeCoin(TChain* chainC){

   int nentriesC=chainC->GetEntries();

    int eventID1 = 0;
    chainC->SetBranchAddress("eventID1",&eventID1);
    double time1 = 0;
    chainC->SetBranchAddress("time1",&time1);
    int eventID2 = 0;
    chainC->SetBranchAddress("eventID2",&eventID2);
    double time2 = 0;
    chainC->SetBranchAddress("time2",&time2);
    int runID    = 0;
    chainC->SetBranchAddress("runID",&runID);

    m_RootTarget->cd();
    TTree * newCoin = chainC->CloneTree(0);
    newCoin->SetAutoSave(2000000000);
    if(m_maxRoot!=0) newCoin->SetMaxTreeSize(m_maxRoot);
    else newCoin->SetMaxTreeSize(17179869184LL);

    // changing CompLevel everywhere
    TBranch *br;
    TIter next(newCoin->GetListOfBranches());
    while ((br=(TBranch*)next())) br->SetCompressionLevel(m_CompLevel);

    int offset      = 0;
    int currentTree = 0;
    float lastRun   = 0;
    float maxtime   =-999999999;
    for(int i=0;i<nentriesC;i++){
       if(chainC->GetEntry(i) <= 0) break; //end of chain
       if(chainC->GetTreeNumber()>currentTree) {
         currentTree++;
         offset+=m_lastEvents[currentTree];
         // check for overlaping time intervalls between different files
         // (not within the same file i.e. no time order assumed)
         if(time1<maxtime) 
           if(m_verboseLevel>0)
               cout<<"Warning - overlapping Coincidences time1 ("
                   <<time1<<") in file: "<<m_vRootFileNames[currentTree].c_str()<<endl;
         if(time2<maxtime) 
           if(m_verboseLevel>0)
               cout<<"Warning - overlapping Coincidences time2 ("
                   <<time2<<") in file: "<<m_vRootFileNames[currentTree].c_str()<<endl;
       }
       // the offset to get a unique event numbering
       if(lastRun!=runID) {
          // run end
          lastRun=runID;
          offset=0; //new run in file we must not change the eventID anymore
       }
       eventID1+=offset;
       eventID2+=offset;
       newCoin->Fill();
       if(maxtime<time1)maxtime=time1;
       if(maxtime<time2)maxtime=time2;
    }
    newCoin->Write();
    delete newCoin;
    return true;
}
/*******************************************************************************************/
