/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateHitFileReader.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "TBranch.h"
#include "GateHit.hh"
#include "GateOutputVolumeID.hh"
#include "GateOutputMgr.hh"
#include "GateTools.hh"
#include "GateHitFileReaderMessenger.hh"
#include "GateHitConvertor.hh"

GateHitFileReader* GateHitFileReader::instance = 0;

// Private constructor: this function should only be called from GetInstance()
GateHitFileReader::GateHitFileReader()
  : GateClockDependent("hitreader",false)
  , m_fileName("gate")
  , m_hitFile(0)
  , m_hitTree(0)
  , m_entries(0)
  , m_currentEntry(0)
{
  // Clear the root-hit structure
  m_hitBuffer.Clear();

  // Create the messenger;
  m_messenger = new GateHitFileReaderMessenger(this);
}





// Public destructor
GateHitFileReader::~GateHitFileReader()
{
  // Clear the file and the queue if it was still open
  TerminateAfterAcquisition();

  // delete the messenger
  delete m_messenger;
}




/* This function allows to retrieve the current instance of the GateHitFileReader singleton

      	If the GateHitFileReader already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateHitFileReader constructor
*/
GateHitFileReader* GateHitFileReader::GetInstance()
{
    if (instance == 0)
      instance = new GateHitFileReader();
    return instance;
}





// This method must be called (normally by the application manager) before starting a new DigiGate acquisition
// It opens the ROOT input file, sets up the hit tree, and loads the first series of hits
void GateHitFileReader::PrepareAcquisition()
{
  // Open the input file
  m_hitFile = new TFile((m_fileName+".root").c_str(),"READ");
  if (!m_hitFile)
	{
		G4String msg = "Could not open the requested hit file '" + m_fileName + ".root'!";
    G4Exception( "GateHitFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg );
	}
  if (!(m_hitFile->IsOpen()))
	{
		G4String msg = "Could not open the requested hit file '" + m_fileName + ".root'!";
    G4Exception( "GateHitFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg );
	}
  // Get the hit tree
  m_hitTree = (TTree*)( m_hitFile->Get(GateHitConvertor::GetOutputAlias()) );
  if (!m_hitTree)
	{
		G4String msg = "Could not find a tree of hits in the ROOT file '" + m_fileName + ".root'!";
    G4Exception( "GateHitFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg);
	}
  // Reset the entry counters
  m_currentEntry=0;
  m_entries = m_hitTree->GetEntries();


  // Set the addresses of the branch buffers: each buffer is a field of the root-hit structure
  GateHitTree::SetBranchAddresses(m_hitTree,m_hitBuffer);

  //  Load the first hit into the root-hit structure
  LoadHitData();
}




/* This method is meant to be called by the primary generator action at the beginning of each event.
   It read a series of hit data from the ROOT file, and stores them into a queue of hits/

   It returns 1 if it managed to read a series of hits for the current run
   It can return 0 in two cases:
   - either it failed to read a series of hits (end-of-file)
   - or the series of hits has a different runID from the current runID, so that this series
     should not be used for the current run but rather for a later run
*/
G4int GateHitFileReader::PrepareNextEvent(G4Event* )
{
  G4cout << " GateHitFileReader::PrepareNextEvent\n";
  // Store the current runID and eventID
  G4int currentEventID = m_hitBuffer.eventID;
  G4int currentRunID = m_hitBuffer.runID;

  // We've reached the end-of-file
  if ( (currentEventID==-1) && (currentRunID==-1) )
    return 0;

  // Load the hits for the current event
  // We loop until the data that have been read are found to be for a different event or run
  while ( (currentEventID == m_hitBuffer.eventID) && (currentRunID == m_hitBuffer.runID) ) {

    // Create a new hit and store it into the hit-queue
    GateHit* aHit =  m_hitBuffer.CreateHit();
    m_hitQueue.push(aHit);

    // Load the next set of hit-data into the root-hit structure
    LoadHitData();
  }

  if (currentRunID==m_hitBuffer.runID){
    // We got a set of hits for the current run -> return 1
    return 1;
  }
  else
  {
    // We got a set of hits for a later run -> return 0
    return 0;
  }
}


// This method is meant to be called by output manager before calling the methods RecordEndOfEvent() of the output modules.
// It creates a new hit-collection, based on the queue of hits previously filled by PrepareNextEvent()
void GateHitFileReader::PrepareEndOfEvent()
{
  // We loop until the hit-queue is empty
  // Each hit is inserted into the crystalSD hit-collection, then removed from the queue
  while (m_hitQueue.size()) {
    GateOutputMgr::GetInstance()->GetHitCollection()->insert(m_hitQueue.front());
    m_hitQueue.pop();
  }
}



// This method must be called (normally by the application manager) after completion of a DigiGate acquisition
// It closes the ROOT input file
void GateHitFileReader::TerminateAfterAcquisition()
{
  // Close the file
  if (m_hitFile) {
    delete m_hitFile;
    m_hitFile=0;
  }

  // If the hit queue was not empty (it should be), clear it up
  while (m_hitQueue.size()) {
    delete m_hitQueue.front();
    m_hitQueue.pop();
  }

  // Note that we don't delete the tree: it was based on the file so
  // I assume it was destroyed at the same time as the file was closed (true?)
  m_hitTree=0;
}



// Reads a set of hit data from the hit-tree, and stores them into the root-hit buffer
void GateHitFileReader::LoadHitData()
{
  // We've reached the end of file: set indicators to tell the caller that the reading failed
  if (m_currentEntry>=m_entries){
    m_hitBuffer.runID=-1;
    m_hitBuffer.eventID=-1;
    return;
  }

  // Read a new set of hit-data: if it failed, set indicators to tell the caller that the reading failed
  if (m_hitTree->GetEntry(m_currentEntry++)<=0) {
    G4cerr << "[GateHitFileReader::LoadHitData]:\n"
      	   << "\tCould not read the next hit!\n";
    m_hitBuffer.runID=-1;
    m_hitBuffer.eventID=-1;
  }
}





/* Overload of the base-class virtual method to print-out a description of the reader

   indent: the print-out indentation (cosmetic parameter)
*/
void GateHitFileReader::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Hit-file name:    " << m_fileName << Gateendl;
  G4cout << GateTools::Indent(indent) << "Hit-file status:  " << (m_hitFile ? "open" : "closed" ) << Gateendl;
  if (m_hitTree) {
    G4cout << GateTools::Indent(indent) << "Hit-tree entries: " << m_entries << Gateendl;
    G4cout << GateTools::Indent(indent) << "Current entry:    " << m_currentEntry << Gateendl;
  }
}


#endif
