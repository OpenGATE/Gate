/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <string>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "GateMergeManager.hh"
using namespace std;

void showhelp(){

 cout<<endl;
 cout<<"  +-------------------------------------------+"<<endl;
 cout<<"  | gjm -- The GATE cluster job output merger |"<<endl;
 cout<<"  +-------------------------------------------+"<<endl;
 cout<<endl;
 cout<<"  Usage: gjm [-options] your_file.split"<<endl;
 cout<<endl;
 cout<<"  You may give the name of the split file created by gjs (see inside the .Gate directory)."<<endl;
 cout<<"  !! This merger is only designed to ROOT output. !!"<<endl;
 cout<<endl;
 cout<<"  Options: "<<endl;
 cout<<"  -outDir path              : where to save the output files default is PWD"<<endl;
// cout<<"  -maxRoot xxx [M|G]        : maximum size for the joint root file. If output size > maxRoot"<<endl;
// cout<<"                              multiple root files will be created: filename_part(n).root "<<endl;
// cout<<"                              xxx is an integer, default unit is M "<<endl;
 cout<<"  -v                        : verbosity 0 1 2 3 - 1 default "<<endl;
 cout<<"  -f                        : forced output - an existing output file will be overwritten"<<endl;
 cout<<"  -cleanonly                : do only a the cleanup step i.e. no merging"<<endl;
 cout<<"                              erase work directory in .Gate and the files from the parallel jobs"<<endl;
 cout<<"  -cleanonlyTest            : just tells you what will be erased by the -cleanonly"<<endl;
 cout<<"  -clean                    : merge and then do the cleanup automatically"<<endl;
 cout<<"  -fastMerge                : correct the output in each file, to be used with a TChain (only for Root output)"<<endl;
 cout<<endl;
 cout<<"  Environment variable: "<<endl;
 cout<<"  GC_DOT_GATE_DIR : points to the .Gate directory"<<endl<<endl;
 exit(0);
}

int main(int argc,char** argv)
{
  string        outDir ="";
  string splitfileName ="";
  Long64_t     maxRoot = 0;
  int     verboseLevel = 1;
  bool          forced = false;
  bool          clean  = false;
  bool          test   = false;
  bool          merge  = true;
  bool       fastMerge = false;

  // Parse the command line
  if (argc==1) showhelp();
  int nextArg=1;
  while (nextArg<argc) { 
    if (!strcmp(argv[nextArg],"-v") && (nextArg+1)<argc){
       nextArg++;
       if(!isdigit(argv[nextArg][0]) ) {
          cout<<"-v "<<argv[nextArg]<<" That's not a number!"<<endl;
          exit(0);
       }
       verboseLevel=atoi(argv[nextArg]);
    } else if (strstr(argv[nextArg],".split") ){
       splitfileName=(string)argv[nextArg];
    } else if (!strcmp(argv[nextArg],"-outDir") && (nextArg+1)<argc){
       nextArg++;
       outDir=argv[nextArg];
    } else if (!strcmp(argv[nextArg],"-f")){
       forced=true;
    } else if (!strcmp(argv[nextArg],"-clean")){
       clean = true;
       merge = true;
       test  = false;
    } else if (!strcmp(argv[nextArg],"-cleanonlyTest")){
       clean = true;
       merge = false;
       test  = true;
    } else if (!strcmp(argv[nextArg],"-fastMerge")){
       fastMerge=true;
    } else if (!strcmp(argv[nextArg],"-cleanonly")){
       clean = true;
       merge = false;
       test  = false;
    } else if (!strcmp(argv[nextArg],"-maxRoot") && (nextArg+1)<argc){
       nextArg++;
       if(!isdigit(argv[nextArg][0]) ) {
          cout<<"-maxRoot "<<argv[nextArg]<<" That's not a number!"<<endl;
          exit(0);
       }
       int fac=0;
       int len=strlen(argv[nextArg]);
       if(len>1){
         if(argv[nextArg][len-1]=='M') fac=1;
         if(argv[nextArg][len-1]=='G') fac=1024;
       }
       if(fac!=0) argv[nextArg][len-1]='\0';
       maxRoot=atol(argv[nextArg]);
       if(fac==0&&nextArg+1<argc){
         if(argv[nextArg+1][0]=='M')      fac=1;
         else if(argv[nextArg+1][0]=='G') fac=1024;
         if(fac!=0) nextArg++;
       }
       if(fac==0){
          fac=1;
          if(verboseLevel>1) cout<<"No unit for maxRoot given -  using M"<<endl;
       }
       maxRoot*=fac*1024*1024-1;
    } else if(!strcmp(argv[nextArg],"-h")
             ||strcmp(argv[nextArg],"-help")) {
       showhelp();
       exit(0);
    } else {
       cout<<"Not a valid argument: "<<argv[nextArg]<<endl;
       showhelp();
       exit(0);
    }
    nextArg++;
  }

  if(outDir!=""){     // check if the outDir exist
     if( outDir.substr(outDir.length()-1,1) != "/" ) outDir+="/";
     std::ifstream dirstream(outDir.c_str());
     if (!dirstream) {
         cout<<"Failed to open output directory "<<outDir<<endl;
         exit(1);
     }
     dirstream.close();
  }

  //create a merge manager
  GateMergeManager* manager = new GateMergeManager(fastMerge,verboseLevel,forced,maxRoot,outDir);

  if(merge) manager->StartMerging(splitfileName);
  if(clean) manager->StartCleaning(splitfileName,test);

  delete manager;
  return 0;
}
