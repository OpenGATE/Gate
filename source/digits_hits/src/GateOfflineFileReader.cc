/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateOfflineFileReader.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "TBranch.h"
#include "GateHit.hh"
#include "GateOutputVolumeID.hh"
#include "GateOutputMgr.hh"
#include "GateTools.hh"
#include "GateOfflineFileReaderMessenger.hh"

GateOfflineFileReader* GateOfflineFileReader::instance = 0;

// Private constructor: this function should only be called from GetInstance()
GateOfflineFileReader::GateOfflineFileReader()
  : GateClockDependent("hitreader",false)
  , m_fileName("gate.root")
  , m_inptuFile(0)
  , m_inputTree(0)
  , m_entries(0)
  , m_currentEntry(0)
  , m_finished(false)
{
  // Clear the root file structure
  m_hitBuffer.Clear();
  m_singleBuffer.Clear();

  // Create the messenger;
  m_messenger = new GateOfflineFileReaderMessenger(this);
}





// Public destructor
GateOfflineFileReader::~GateOfflineFileReader()
{
  // Clear the file and the queue if it was still open
  TerminateAfterAcquisition();

  // delete the messenger
  delete m_messenger;
}




/* This function allows to retrieve the current instance of the GateOfflineFileReader singleton

      	If the GateOfflineFileReader already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateOfflineFileReader constructor
*/
GateOfflineFileReader* GateOfflineFileReader::GetInstance()
{
    if (instance == 0)
      instance = new GateOfflineFileReader();
    return instance;
}





// This method must be called (normally by the application manager) before starting a new DigiGate acquisition
void GateOfflineFileReader::PrepareAcquisition()
{

	//G4cout<<"GateOfflineFileReader::PrepareAcquisition "<< G4endl;
	 m_finished = false;
  // Open the input file

  m_inptuFile = TFile::Open(m_fileName.c_str(), "READ");

  if (!m_inptuFile || m_inptuFile->IsZombie())
  {
      G4String msg =
          "Could not open hit file '" + m_fileName + "'";
      G4Exception("GateOfflineFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg);
  }
  // Get the hit tree
  if (GateOutputMgr::GetInstance()->GetDigiMode()==kofflineMode )
    {
	  m_inputTree = (TTree*)( m_inptuFile->Get("Hits")) ;
    }
    else
    {
      m_inputTree = (TTree*)( m_inptuFile->Get("Singles")) ;
    }

// G4cout<<"m_inputTree "<<m_inputTree<< G4endl;
  //m_inputTree = (TTree*)( m_inptuFile->Get("Hits")) ;

  if (!m_inputTree)
	{
		G4String msg = "Could not find a tree of hits in the ROOT file '" + m_fileName + "'!";
    G4Exception( "GateOfflineFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg);
	}
  // Reset the entry counters
  m_currentEntry=0;
  m_entries = m_inputTree->GetEntries();


  // Set the addresses of the branch buffers: each buffer is a field of the root-hit structure
  if (GateOutputMgr::GetInstance()->GetDigiMode()==kofflineMode)
	  GateHitTree::SetBranchAddresses(m_inputTree,m_hitBuffer);
  else
  	 GateSingleTree::SetBranchAddresses(m_inputTree,m_singleBuffer);


  //GateHitTree::SetBranchAddresses(m_inputTree,m_hitBuffer);

  //  Load the first hit into the root-hit structure
  if (GateOutputMgr::GetInstance()->GetDigiMode()==kofflineMode)
 	  LoadHitData();
  else
	  LoadSinglesData();


}




/* This method is meant to be called by the primary generator action at the beginning of each event.
   It read a series of hit data from the ROOT file, and stores them into a queue of hits/

   It returns 1 if it managed to read a series of hits for the current run
   It can return 0 in two cases:
   - either it failed to read a series of hits (end-of-file)
   - or the series of hits has a different runID from the current runID, so that this series
     should not be used for the current run but rather for a later run
*/
G4int GateOfflineFileReader::PrepareNextEventFromHits(G4Event* )
{

	//G4cout<<"GateOfflineFileReader::PrepareNextEvent "<<G4endl;

  for (auto hit : m_hitVector)
  	    delete hit;

  	m_hitVector.clear();

  // Store the current runID and eventID
  G4int currentEventID = m_hitBuffer.eventID;
  G4int currentRunID = m_hitBuffer.runID;

  //G4cout<<"currentEventID "<<currentEventID<<G4endl;
  // We've reached the end-of-file
  if ( (currentEventID==-1) && (currentRunID==-1) )
  {
        m_finished = true;
        return 0;
    }

  // Load the hits for the current event
  // We loop until the data that have been read are found to be for a different event or run
  while ( (currentEventID == m_hitBuffer.eventID) && (currentRunID == m_hitBuffer.runID) ) {

    // Create a new hit and store it into the hit-queue
    GateHit* aHit =  m_hitBuffer.CreateHit();
    m_hitVector.push_back(aHit);

    // Load the next set of hit-data into the root-hit structure
    LoadHitData();
  }

  if (!m_hitVector.empty())
        return 1;
    return 0;
}



G4int GateOfflineFileReader::PrepareNextEventFromSingles(G4Event* )
{

	//G4cout<<"PrepareNextEventFromSingles::PrepareNextEvent "<<G4endl;

  for (auto digi : m_digiVector)
  	    delete digi;

  	m_digiVector.clear();

  // Store the current runID and eventID
  G4int currentEventID = m_singleBuffer.eventID;
  G4int currentRunID = m_singleBuffer.runID;

  //G4cout<<"currentEventID "<<currentEventID<<G4endl;
  // We've reached the end-of-file
  if ( (currentEventID==-1) && (currentRunID==-1) )
  {
        m_finished = true;
        return 0;
    }

  // Load the hits for the current event
  // We loop until the data that have been read are found to be for a different event or run
  while ( (currentEventID == m_singleBuffer.eventID) && (currentRunID == m_singleBuffer.runID) ) {

    // Create a new hit and store it into the hit-queuet
	  GateDigi* aDigi=  m_singleBuffer.CreateDigi();
	  m_digiVector.push_back(aDigi);

    // Load the next set of hit-data into the root-hit structure
    LoadSinglesData();
  }

  if (!m_digiVector.empty())
        return 1;
    return 0;
}


// This method must be called (normally by the application manager) after completion of a DigiGate acquisition
// It closes the ROOT input file
void GateOfflineFileReader::TerminateAfterAcquisition()
{
  // Close the file
  if (m_inptuFile) {
    delete m_inptuFile;
    m_inptuFile=0;
  }

  if (GateOutputMgr::GetInstance()->GetDigiMode()==kofflineMode )
     {
	  for (auto hit : m_hitVector)
	        delete hit;
	    m_hitVector.clear();
     }
     else
     {
      for (auto digi : m_digiVector)
    	  delete digi;
      m_digiVector.clear();
     }


  // Note that we don't delete the tree: it was based on the file so
  // I assume it was destroyed at the same time as the file was closed (true?)
  m_inputTree=0;
}



// Reads a set of hit data from the hit-tree, and stores them into the root-hit buffer
void GateOfflineFileReader::LoadHitData()
{

	//G4cout<<"GateOfflineFileReader::LoadHitData "<< m_entries <<G4endl;
  // We've reached the end of file: set indicators to tell the caller that the reading failed
  if (m_currentEntry>=m_entries){
    m_hitBuffer.runID=-1;
    m_hitBuffer.eventID=-1;
    return;
  }

  // Read a new set of hit-data: if it failed, set indicators to tell the caller that the reading failed
  if (m_inputTree->GetEntry(m_currentEntry++)<=0) {
    G4cerr << "[GateOfflineFileReader::LoadHitData]:\n"
      	   << "\tCould not read the next hit!\n";
    m_hitBuffer.runID=-1;
    m_hitBuffer.eventID=-1;
  }
}

// Reads a set of hit data from the hit-tree, and stores them into the root-hit buffer
void GateOfflineFileReader::LoadSinglesData()
{

	//G4cout<<"GateOfflineFileReader::LoadSinglesData "<< m_entries <<G4endl;
  // We've reached the end of file: set indicators to tell the caller that the reading failed
  if (m_currentEntry>=m_entries){
    m_singleBuffer.runID=-1;
    m_singleBuffer.eventID=-1;
    return;
  }

  // Read a new set of Singles-data: if it failed, set indicators to tell the caller that the reading failed
  if (m_inputTree->GetEntry(m_currentEntry++)<=0) {
    G4cerr << "[GateOfflineFileReader::LoadSinglesData]:\n"
      	   << "\tCould not read the next Singles!\n";
    m_singleBuffer.runID=-1;
    m_singleBuffer.eventID=-1;
  }
}



/* Overload of the base-class virtual method to print-out a description of the reader

   indent: the print-out indentation (cosmetic parameter)
*/
void GateOfflineFileReader::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Input file name:    " << m_fileName << Gateendl;
  G4cout << GateTools::Indent(indent) << "Input file status:  " << (m_inptuFile ? "open" : "closed" ) << Gateendl;
  if (m_inputTree) {
    G4cout << GateTools::Indent(indent) << "Input tree entries: " << m_entries << Gateendl;
    G4cout << GateTools::Indent(indent) << "Current entry:    " << m_currentEntry << Gateendl;
  }
}


#endif
