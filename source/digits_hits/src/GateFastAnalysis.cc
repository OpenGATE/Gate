/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*  Update: V. Cuplov   15 Feb. 2012
            New output file (ntuple) dedicated to the Optical Photon Validation. 
*/

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
#include "G4RunManager.hh"
#include "GateCrystalHit.hh"
#include "GateSourceMgr.hh"
#include "GateOutputMgr.hh"

// v. cuplov 15.02.12 
#include "G4OpticalPhoton.hh"
#include "GateTrajectoryNavigator.hh"
// v. cuplov 15.02.12

GateFastAnalysis::GateFastAnalysis(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode) 
  : GateVOutputModule(name,outputMgr,digiMode)
{
  m_messenger = new GateFastAnalysisMessenger(this);
  SetVerboseLevel(0);
  // this module is disabled by default
  Enable(false);

// v. cuplov 15.02.12
  m_opticalfile = 0;
  m_trajectoryNavigator = new GateTrajectoryNavigator();
// v. cuplov 15.02.12

}

GateFastAnalysis::~GateFastAnalysis() 
{
  delete m_messenger;
  if (nVerboseLevel > 0) G4cout << "GateFastAnalysis deleting..." << G4endl;

// v. cuplov 15.02.12
    delete m_trajectoryNavigator;
// v. cuplov 15.02.12

}

const G4String& GateFastAnalysis::GiveNameOfFile()
{
  m_noFileName = "  "; // 2 spaces for output module with no fileName
  return m_noFileName;
}

void GateFastAnalysis::RecordBeginOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordBeginOfAcquisition" << G4endl;

// v. cuplov 15.02.12 - Here you can add new leaves to the ntuple
  m_opticalfile = new TFile("ValidateOpticalPhotons.root" ,"RECREATE","ROOT file with Event Data");
  OpticalTuple = new TTree(G4String("EventData").c_str(),"EventData");
  OpticalTuple->Branch(G4String("NumCrystalTransport").c_str(),&nCrystalTransport,"nCrystalTransport/I");
  OpticalTuple->Branch(G4String("NumCrystalOptAbs").c_str(),&nCrystalOpticalAbsorption,"nCrystalOpticalAbsorption/I");
  OpticalTuple->Branch(G4String("NumCrystalOptRay").c_str(),&nCrystalOpticalRayleigh,"nCrystalOpticalRayleigh/I");
  OpticalTuple->Branch(G4String("NumCrystalOptMie").c_str(),&nCrystalOpticalMie,"nCrystalOpticalMie/I");
  OpticalTuple->Branch(G4String("NumPhantomTransport").c_str(),&nPhantomTransport,"nPhantomTransport/I");
  OpticalTuple->Branch(G4String("NumPhantomOptAbs").c_str(),&nPhantomOpticalAbsorption,"nPhantomOpticalAbsorption/I");
  OpticalTuple->Branch(G4String("NumPhantomOptRay").c_str(),&nPhantomOpticalRayleigh,"nPhantomOpticalRayleigh/I");
  OpticalTuple->Branch(G4String("NumPhantomOptMie").c_str(),&nPhantomOpticalMie,"nPhantomOpticalMie/I");
  OpticalTuple->Branch(G4String("PhantomOpticalAbsorption_x").c_str(),&PhantomOpticalAbsorption_x,"PhantomOpticalAbsorption_x/D");
  OpticalTuple->Branch(G4String("PhantomOpticalAbsorption_y").c_str(),&PhantomOpticalAbsorption_y,"PhantomOpticalAbsorption_y/D");
  OpticalTuple->Branch(G4String("PhantomOpticalAbsorption_z").c_str(),&PhantomOpticalAbsorption_z,"PhantomOpticalAbsorption_z/D");
  OpticalTuple->Branch(G4String("CrystalOpticalAbsorption_x").c_str(),&CrystalOpticalAbsorption_x,"CrystalOpticalAbsorption_x/D");
  OpticalTuple->Branch(G4String("CrystalOpticalAbsorption_y").c_str(),&CrystalOpticalAbsorption_y,"CrystalOpticalAbsorption_y/D");
  OpticalTuple->Branch(G4String("CrystalOpticalAbsorption_z").c_str(),&CrystalOpticalAbsorption_z,"CrystalOpticalAbsorption_z/D");
  OpticalTuple->Branch(G4String("CrystalOpticalPhoton_x").c_str(),&CrystalOpticalPhoton_x,"CrystalOpticalPhoton_x/D");
  OpticalTuple->Branch(G4String("CrystalOpticalPhoton_y").c_str(),&CrystalOpticalPhoton_y,"CrystalOpticalPhoton_y/D");
  OpticalTuple->Branch(G4String("CrystalOpticalPhoton_z").c_str(),&CrystalOpticalPhoton_z,"CrystalOpticalPhoton_z/D");
  OpticalTuple->Branch(G4String("PhantomOpticalPhoton_x").c_str(),&PhantomOpticalPhoton_x,"PhantomOpticalPhoton_x/D");
  OpticalTuple->Branch(G4String("PhantomOpticalPhoton_y").c_str(),&PhantomOpticalPhoton_y,"PhantomOpticalPhoton_y/D");
  OpticalTuple->Branch(G4String("PhantomOpticalPhoton_z").c_str(),&PhantomOpticalPhoton_z,"PhantomOpticalPhoton_z/D");
// v. cuplov 15.02.12

}

void GateFastAnalysis::RecordEndOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordEndOfAcquisition" << G4endl;

// v. cuplov 15.02.12
  if ( OpticalTuple != 0 ) {OpticalTuple->Print();}
  m_opticalfile = OpticalTuple->GetCurrentFile();
  m_opticalfile->Write();
  if ( m_opticalfile->IsOpen() ){ m_opticalfile->Close();}
// v. cuplov 15.02.12


}

void GateFastAnalysis::RecordBeginOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordBeginOfRun" << G4endl;
}

void GateFastAnalysis::RecordEndOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordEndOfRun" << G4endl;
}

void GateFastAnalysis::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordBeginOfEvent" << G4endl;
}

void GateFastAnalysis::RecordEndOfEvent(const G4Event* event)
{
// !!! New !!! NTUPLE
// v. cuplov 15.02.12
  G4TrajectoryContainer* trajectoryContainer = event->GetTrajectoryContainer(); 
  if (trajectoryContainer)  m_trajectoryNavigator->SetTrajectoryContainer(trajectoryContainer);

   GatePhantomHitsCollection* PHC = GetOutputMgr()->GetPhantomHitCollection();
   GateCrystalHitsCollection* CHC = GetOutputMgr()->GetCrystalHitCollection();

        nPhantomOpticalRayleigh = 0;
        nPhantomOpticalMie = 0;
        nPhantomOpticalAbsorption = 0;
        nPhantomTransport = 0;
        nCrystalOpticalRayleigh = 0;
        nCrystalOpticalMie = 0;
        nCrystalOpticalAbsorption = 0;
        nCrystalTransport = 0;

// Looking at Phantom Hit Collection:
   if (PHC) {
         G4int NpHits = PHC->entries();    
         for (G4int iPHit=0;iPHit<NpHits;iPHit++)
            {
               if ((*CHC)[iPHit]->GoodForAnalysis())
                {
                GatePhantomHit* pHit = (*PHC)[iPHit];
                G4String processName = (*PHC)[iPHit]->GetProcess();

                PhantomOpticalPhoton_x = pHit->GetPos().x(); // record phantom optical photon hit x-position
                PhantomOpticalPhoton_y = pHit->GetPos().y(); // record phantom optical photon hit y-position
                PhantomOpticalPhoton_z = pHit->GetPos().z(); // record phantom optical photon hit z-position

                if (processName.find("OpRayleigh") != G4String::npos)  nPhantomOpticalRayleigh++;
                if (processName.find("OpticalMie") != G4String::npos)  nPhantomOpticalMie++;
                if (processName.find("OpticalAbsorption") != G4String::npos) {

                       nPhantomOpticalAbsorption++;
                       PhantomOpticalAbsorption_x = pHit->GetPos().x();  // record MEDIUM ABSORBED optical photon hit x-position in phantom
                       PhantomOpticalAbsorption_y = pHit->GetPos().y();  // record MEDIUM ABSORBED optical photon hit y-position in phantom
                       PhantomOpticalAbsorption_z = pHit->GetPos().z();  // record MEDIUM ABSORBED optical photon hit z-position in phantom

                     }

                if (processName.find("Transport") != G4String::npos) nPhantomTransport++;

                }  // end GoodForAnalysis()
              } // end loop over phantom hits
            } // end if PHC


// Looking at Crystal Hits Collection:
  if (CHC) {    
        G4int NbHits = CHC->entries();  

    G4int sourceID = (((GateSourceMgr::GetInstance())->GetSourcesForThisEvent())[0])->GetSourceID();
    G4int eventID  = event->GetEventID();
    G4int runID    = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();

        for (G4int iHit=0;iHit<NbHits;iHit++) 
           {
              if ((*CHC)[iHit]->GoodForAnalysis())
               {
               GateCrystalHit* aHit = (*CHC)[iHit];
               G4String processName = aHit->GetProcess();

                CrystalOpticalPhoton_x = aHit->GetGlobalPos().x(); // record crystal optical photon hit x-position
                CrystalOpticalPhoton_y = aHit->GetGlobalPos().y(); // record crystal optical photon hit y-position
                CrystalOpticalPhoton_z = aHit->GetGlobalPos().z(); // record crystal optical photon hit z-position

                if (processName.find("OpRayleigh") != G4String::npos)  nCrystalOpticalRayleigh++;
                if (processName.find("OpticalMie") != G4String::npos)  nCrystalOpticalMie++;
                if (processName.find("OpticalAbsorption") != G4String::npos) {

                       nCrystalOpticalAbsorption++;
                       CrystalOpticalAbsorption_x = aHit->GetGlobalPos().x();  // record MEDIUM ABSORBED optical photon hit x-position in crystal
                       CrystalOpticalAbsorption_y = aHit->GetGlobalPos().y();  // record MEDIUM ABSORBED optical photon hit y-position in crystal
                       CrystalOpticalAbsorption_z = aHit->GetGlobalPos().z();  // record MEDIUM ABSORBED optical photon hit z-position in crystal

                    }

                if (processName.find("Transportation") != G4String::npos) nCrystalTransport++;

// v. cuplov 15.02.12

// This is the old code relica
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
// End of old code relica

                } // end GoodForAnalysis()
            } // end loop over crystal hits
  } // end if CHC

  if (!trajectoryContainer) {

//  G4cerr << "GateToRoot::RecordEndOfEvent : ERROR : NULL trajectoryContainer!" << G4endl;

           } else {

  OpticalTuple->Fill();
}

 if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordEndOfEvent" << G4endl;

}

void GateFastAnalysis::RecordStepWithVolume(const GateVVolume *, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateFastAnalysis::RecordStep" << G4endl;
}

void GateFastAnalysis::SetVerboseLevel(G4int val) 
{ nVerboseLevel = val;}

#endif
