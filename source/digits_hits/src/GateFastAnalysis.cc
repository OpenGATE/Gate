/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateFastAnalysis.hh"
#include "GateFastAnalysisMessenger.hh"
#include "GateVVolume.hh"

#include "globals.hh"

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4HCofThisEvent.hh"
#include "GateHit.hh"
#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"
#include "GateDigitizerMgr.hh"

GateFastAnalysis::GateFastAnalysis(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
{
  m_messenger = new GateFastAnalysisMessenger(this);
  SetVerboseLevel(0);
  // this module is disabled by default
  Enable(false);
}

GateFastAnalysis::~GateFastAnalysis()
{
  delete m_messenger;
  if (nVerboseLevel > 0) G4cout << "GateFastAnalysis deleting...\n";
}

const G4String& GateFastAnalysis::GiveNameOfFile()
{
  m_noFileName = "  "; // 2 spaces for output module with no fileName
  return m_noFileName;
}

void GateFastAnalysis::RecordBeginOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordBeginOfAcquisition\n";
}

void GateFastAnalysis::RecordEndOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordEndOfAcquisition\n";
}

void GateFastAnalysis::RecordBeginOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordBeginOfRun\n";
}

void GateFastAnalysis::RecordEndOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordEndOfRun\n";
}

void GateFastAnalysis::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordBeginOfEvent\n";

  //GateDigitizerMgr* digitizerMgr=GateDigitizerMgr::GetInstance();
  // digitizerMgr->m_alreadyRun=false;
}

void GateFastAnalysis::RecordEndOfEvent(const G4Event* event)
{
	 if (nVerboseLevel > 2)
	    G4cout << "GateFastAnalysis::RecordEndOfEvent\n";

	//OK GND 2022
	std::vector<GateHitsCollection*> CHC_vector = GetOutputMgr()->GetHitCollections();
	for (size_t i=0; i<CHC_vector.size();i++ )
	   {
		   GateHitsCollection* CHC = CHC_vector[i];
	//*OK GND 2022

	// Looking at Crystal Hits Collection:
	  if (CHC) {
			G4int NbHits = CHC->entries();

		G4int sourceID = (((GateSourceMgr::GetInstance())->GetSourcesForThisEvent())[0])->GetSourceID();
		G4int eventID  = event->GetEventID();
		G4int runID    = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();

			for (G4int iHit=0;iHit<NbHits;iHit++)
			   {
				  if ((*CHC)[iHit]->GoodForAnalysis())
				   {
				   GateHit* aHit = (*CHC)[iHit];
				   G4String processName = aHit->GetProcess();

		(*CHC)[iHit]->SetSourceID(sourceID);
		(*CHC)[iHit]->SetEventID(eventID);
		(*CHC)[iHit]->SetRunID(runID);
		// the following parameters are not calculated and are therefore set to -1
		// or "NULL"  to indicate no value
			G4ThreeVector sourcePosition(-1,-1,-1);
			(*CHC)[iHit]->SetSourcePosition(sourcePosition);
		(*CHC)[iHit]->SetNPhantomCompton(-1);
		(*CHC)[iHit]->SetNPhantomRayleigh(-1);
		(*CHC)[iHit]->SetComptonVolumeName("NULL");
		(*CHC)[iHit]->SetRayleighVolumeName("NULL");
		(*CHC)[iHit]->SetPhotonID(-1);
		(*CHC)[iHit]->SetPrimaryID(-1);
		(*CHC)[iHit]->SetNCrystalCompton(-1);
		(*CHC)[iHit]->SetNCrystalRayleigh(-1);

					} // end GoodForAnalysis()
				} // end loop over crystal hits
	  } // end if CHC
  }//end of loop over hits collections

	//OK GND 2022
	GateDigitizerMgr* digitizerMgr=GateDigitizerMgr::GetInstance();

	 if(!digitizerMgr->m_alreadyRun)
		 {

		 if (digitizerMgr->m_recordSingles|| digitizerMgr->m_recordCoincidences)
				 {

				 	 digitizerMgr->RunDigitizers();
				 	 digitizerMgr->RunCoincidenceSorters();
				 	 digitizerMgr->RunCoincidenceDigitizers();
				 }
		 }
}

void GateFastAnalysis::RecordStepWithVolume(const GateVVolume *, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordStep\n";
}

void GateFastAnalysis::SetVerboseLevel(G4int val)
{ nVerboseLevel = val;}

#endif
