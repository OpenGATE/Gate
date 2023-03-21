/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateCCSinglesFileReader.hh"

//#ifdef G4ANALYSIS_USE_ROOT

#include <TBranch.h>
#include "GateOutputVolumeID.hh"
#include "GateTools.hh"
#include "GateCCRootDefs.hh"
#include "GateDigi.hh"

GateCCSinglesFileReader* GateCCSinglesFileReader::instance = 0;



// Private constructor: this function should only be called from GetInstance()
GateCCSinglesFileReader::GateCCSinglesFileReader(G4String file)
  : m_singlesFile(0)
  , m_singlesTree(0)
  , m_entries(0)
  , m_currentEntry(0)
{


  unsigned lastPointPos=file.rfind(".");
  std::string pathExt = file.substr(lastPointPos);
  m_fileName=file.substr(0,lastPointPos);
  unsigned lastDirect=file.rfind("/");
  m_filePath= file.substr(0,lastDirect+1);

  std::string nameExt=file.substr(lastDirect+1);
  //m_fileName=nameExt.substr(0,lastDot);
  //std::cout<<"directory "<<m_filePath<<std::endl;
  std::cout<<"name "<<m_fileName<<std::endl;
 // Clear the root-Singles structure
  m_singleBuffer.Clear();

  pList=new GatePulseList("layers");

}




// Public destructor
GateCCSinglesFileReader::~GateCCSinglesFileReader()
{
  // Clear the file and the queue if it was still open
  TerminateAfterAcquisition();

}





GateCCSinglesFileReader* GateCCSinglesFileReader::GetInstance(G4String filename)
{
  if (instance == 0)
    instance = new GateCCSinglesFileReader(filename);
  return instance;
}




void GateCCSinglesFileReader::PrepareAcquisition()
{
    std::cout<<"Preparing acquisition"<<std::endl;
  // Open the input file
  m_singlesFile = new TFile((m_fileName+".root").c_str(),"READ");
  if (!m_singlesFile)
    {
      G4String msg = "Could not open the requested Singles file '" + m_fileName + ".root'!";
      G4Exception( "GateCCSinglesFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg );
    }
  if (!(m_singlesFile->IsOpen()))
    {
      G4String msg = "Could not open the requested Singles file '" + m_fileName + ".root'!";
      G4Exception( "GateCCSinglesFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg );
    }
  // Get the Singles tree
  m_singlesTree = (TTree*)( m_singlesFile->Get("Singles") );
  if (!m_singlesTree)
    {
      G4String msg = "Could not find a tree of Singles in the ROOT file '" + m_fileName + ".root'!";
      G4Exception( "GateCCSinglesFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg);
    }
  // Reset the entry counters
  m_currentEntry=0;
  m_entries = m_singlesTree->GetEntries();
  //std::cout<<"entries Singles root "<<m_entries<<std::endl;

  //std::cout<<"setting branch address"<<std::endl;
  // Set the addresses of the branch buffers: each buffer is a field of the root-Singles structure
  GateCCSingleTree::SetBranchAddresses(m_singlesTree,m_singleBuffer);



 //std::cout<<"load data"<<std::endl;
  //  Load the first Singles into the root-Singles structure
  LoadSinglesData();
  std::cout<<"Done"<<std::endl;
}




/*

   It returns 1 if it managed to read a series of Singless for the current run
   It can return 0 in two cases:
   - either it failed to read a series of Singless (end-of-file)
   - or the series of Singless has a different runID from the current runID, so that this series
   should not be used for the current run but rather for a later run
*/
G4int GateCCSinglesFileReader::PrepareNextEvent( )
{
  //G4cout << " GateCCSinglesFileReader::PrepareNextEvent\n";
  // Store the current runID and eventID
  G4int currentEventID = m_singleBuffer.eventID;
  G4int currentRunID = m_singleBuffer.runID;
   //G4cout<<"plist antes de erase size="<<pList->size()<<G4endl;
 //pList->empty();
   if(pList->size()>0){
   pList->erase(pList->begin(),pList->end());
   }
  // G4cout<<"plist size="<<pList->size()<<G4endl;

  // We've reached the end-of-file
  if ( (currentEventID==-1) && (currentRunID==-1) )
    return 0;

  // Load the Singless for the current event
  // We loop until the data that have been read are found to be for a different event or run
  while ( (currentEventID == m_singleBuffer.eventID) && (currentRunID == m_singleBuffer.runID) ) {
      // Create a new Singles and store it into the Singles-queue

        GateDigi* aSingleDigi=m_singleBuffer.CreateSingle();
         // OK GND 2022 TODO: uncomment for CC when no pulse list
        // pList->push_back(&aSingleDigi->GetPulse());

      // Load the next set of Singles-data into the root-Singles structure
      LoadSinglesData();
       if ( (currentEventID==-1) && (currentRunID==-1) ) break;
  }

  if (currentRunID==m_singleBuffer.runID){
    // We got a set of Singless for the current run -> return 1
    return 1;
  }
  else
    {
      // We got a set of Singless for a later run -> return 0
      return 0;
    }
}


GatePulseList*  GateCCSinglesFileReader::PrepareEndOfEvent()
{

  return pList;
}




void GateCCSinglesFileReader::TerminateAfterAcquisition()
{
  // Close the file
  if (m_singlesFile) {
    delete m_singlesFile;
    m_singlesFile=0;
  }



  // Note that we don't delete the tree: it was based on the file so
  // I assume it was destroyed at the same time as the file was closed (true?)
  m_singlesTree=0;
}


G4bool GateCCSinglesFileReader::HasNextEvent(){

  if (m_currentEntry>=m_entries){

    return false;
  }
  else{
    return true;
  }
}

// Reads a set of Singles data from the Singles-tree, and stores them into the root-Singles buffer
void GateCCSinglesFileReader::LoadSinglesData()
{
  // We've reached the end of file: set indicators to tell the caller that the reading failed
  if (m_currentEntry>=m_entries){
    m_singleBuffer.runID=-1;
    m_singleBuffer.eventID=-1;
    return;
  }

  // Read a new set of Singles-data: if it failed, set indicators to tell the caller that the reading failed
  if (m_singlesTree->GetEntry(m_currentEntry++)<=0) {
    G4cerr << "[GateCCSinglesFileReader::LoadSinglesData]:\n"
           << "\tCould not read the next Singles!\n";
    m_singleBuffer.runID=-1;
    m_singleBuffer.eventID=-1;
  }
}





/* Overload of the base-class virtual method to print-out a description of the reader

   indent: the print-out indentation (cosmetic parameter)
*/
void GateCCSinglesFileReader::Describe(size_t indent)
{
 // GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Singles-file name:    " << m_fileName << Gateendl;
  G4cout << GateTools::Indent(indent) << "Singles-file status:  " << (m_singlesFile ? "open" : "closed" ) << Gateendl;
  if (m_singlesTree) {
    G4cout << GateTools::Indent(indent) << "Singles-tree entries: " << m_entries << Gateendl;
    G4cout << GateTools::Indent(indent) << "Current entry:    " << m_currentEntry << Gateendl;
  }
}


//#endif
