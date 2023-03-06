/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFDataToRoot.hh"
#include "GateARFDataToRootMessenger.hh"
#include "GateVGeometryVoxelStore.hh"

#include "globals.hh"

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "GateHit.hh"
#include "GatePhantomHit.hh"
#include "G4VHitsCollection.hh"

#include "G4Trajectory.hh"

#include "G4VProcess.hh"
#include "G4ios.hh"
#include <iomanip>
#include "G4UImanager.hh"
#include "GatePrimaryGeneratorAction.hh"
//#include "GateHitConvertor.hh"

#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"
#include "G4Gamma.hh"
#include "GateApplicationMgr.hh"

#include "GateDigitizer.hh"
#include "GateDigi.hh"
#include "GateOutputMgr.hh"
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "G4DigiManager.hh"

GateARFDataToRoot::GateARFDataToRoot(const G4String& name,
                                     GateOutputMgr* outputMgr,
                                     DigiMode digiMode) :
    GateVOutputModule(name, outputMgr, digiMode)
  {
  m_isEnabled = false; /* Keep this flag false: all output are disabled by default
   Moreover this output will slow down a lot all the simulation ! */
  nVerboseLevel = 0;
  mRootMessenger = new GateARFDataToRootMessenger(this);
  mDrfProjectionMode = 0;
  mArfDataFilename = " "; /* All default output file from all output modules are set to " ".
   They are then checked in GateApplicationMgr::StartDAQ, using
   the VOutputModule pure virtual method GiveNameOfFile() */
  mArfDataFile = 0;
  mArfDataTree = 0;
  mXPlane = 0.;
  mNbofGoingOutPhotons = 0;
  mNbofStraightPhotons = 0;
  mNbofGoingInPhotons = 0;
  mNbofKilledInsideCrystalPhotons = 0;
  mNbofKilledInsideColliPhotons = 0;
  mNbofKilledInsideCamera = 0;
  mNbOfSourcePhotons = 0;
  mNbofStoredPhotons = 0;
  mNbOfHeads = 0;
  mInCamera = 0;
  mOutCamera = 0;
  }

GateARFDataToRoot::~GateARFDataToRoot()
  {
  delete mRootMessenger;
  if (nVerboseLevel > 0)
    {
    G4cout << "GateARFDataToRoot deleting...\n";
    }

  }

const G4String& GateARFDataToRoot::GiveNameOfFile()
  {
  return mArfDataFilename;
  }

/* Method called at the beginning of each acquisition by the application manager: opens the ROOT file and prepare the trees */
void GateARFDataToRoot::RecordBeginOfAcquisition()
  {
  mArfDataFile = new TFile(mArfDataFilename.c_str(), "RECREATE", "ROOT file for ARF purpose");
  mArfDataTree = new TTree("theTree", "ARF Data Tree");
  mArfDataTree->Branch("Edep", &mArfData.mDepositedEnergy, "Edep/D");
  mArfDataTree->Branch("outY", &mArfData.mProjectionPositionY, "outY/D");
  mArfDataTree->Branch("outX", &mArfData.mProjectionPositionX, "outX/D");
  mNbOfPhotonsTree = new TTree("theNumberOfPhoton", " statistics on Simulated Photons");
  mNbOfPhotonsTree->Branch("NOfOutGoingPhot", &mNbofGoingOutPhotons, "NbOfOutGoingPhotons/l");
  mNbOfPhotonsTree->Branch("NbOfInGoingPhot", &mNbofGoingInPhotons, "NbOfInGoingPhotons/l");
  mNbOfPhotonsTree->Branch("NbOfSourcePhot", &mNbOfSourcePhotons, "NbOfSourcePhotons/l");
  mNbOfPhotonsTree->Branch("NbOfInCameraPhot", &mInCamera, "NbOfInCameraPhotons/l");
  mNbOfPhotonsTree->Branch("NbOfOutCameraPhot", &mOutCamera, "NbOfOutCameraPhotons/l");
  mNbOfPhotonsTree->Branch("NbOfStoredPhotons", &mNbofStoredPhotons, "NbOfStoredPhotons/l");
  mNbOfPhotonsTree->Branch("NbOfHeads", &mNbOfHeads, "NbOfHeads/I");
  }

void GateARFDataToRoot::RecordEndOfAcquisition()
  {
  CloseARFDataRootFile();
  DisplayARFStatistics();
  }

void GateARFDataToRoot::RecordBeginOfRun(const G4Run*)
  {
  }

void GateARFDataToRoot::RecordEndOfRun(const G4Run*)
  {
  }

void GateARFDataToRoot::RecordBeginOfEvent(const G4Event*)
  {

  }

void GateARFDataToRoot::RecordEndOfEvent(const G4Event* event)
  {
  RecordDigitizer(event);
  }

void GateARFDataToRoot::RecordDigitizer(const G4Event*)
  {
  if (nVerboseLevel > 2)
    {
    G4cout << "GateARFDataToRoot::RecordDigitizer -- begin \n";
    }
  /* Get Digitizer information */
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  //OK GND 2022
  G4int collectionID = GetCollectionID(mSingleDigiCollectionName); //fDM->GetDigiCollectionID(mSingleDigiCollectionName);
  //G4int collectionID = fDM->GetDigiCollectionID(mSingleDigiCollectionName);
  const GateDigiCollection * SDC = (GateDigiCollection*) (fDM->GetDigiCollection(collectionID));

  if (!SDC)
    {
    if (nVerboseLevel > 0)
      {
      G4cout << "[GateARFDataToRoot::SingleOutputChannel::RecordDigitizer]:"
             << " digi collection '"
             << mSingleDigiCollectionName
             << "' not found\n";
      }
    }
  else
    {
    /* Digi loop */

    if (nVerboseLevel > 0)
      {
      G4cout << "[GateARFDataToRoot::SingleOutputChannel::RecordDigitizer]: Total Digits : "
             << SDC->entries()
             << " in digi collection '"
             << mSingleDigiCollectionName
             << "' \n";
      }
    G4int nDigi = SDC->entries();
    for (G4int digi = 0; digi < nDigi; digi++)
      {
      StoreARFData((*SDC)[digi]);
      } /* we store the ARF data */

    }
  if (nVerboseLevel > 2)
    {
    G4cout << "GateARFDataToRoot::RecordDigitizer -- end \n";
    }

  }

void GateARFDataToRoot::RegisterNewSingleDigiCollection(const G4String& aCollectionName, G4bool)
  {
  mSingleDigiCollectionName = aCollectionName;
  }

void GateARFDataToRoot::IncrementNbOfSourcePhotons()
  {
  mNbOfSourcePhotons++;
  if (nVerboseLevel > 1 && (mNbOfSourcePhotons % 1000000) == 0)
    {
    DisplayARFStatistics();
    }
  }

void GateARFDataToRoot::SetARFDataRootFileName(G4String aName)
  {
  mArfDataFilename = aName + ".root";
  }

void GateARFDataToRoot::CloseARFDataRootFile()
  {
  mArfDataFile = mArfDataTree->GetCurrentFile();
  if (mArfDataFile->IsOpen())
    {
    mNbOfPhotonsTree->Fill();
    mArfDataFile->Write();
    mArfDataFile->Close();
    }
  }

G4int GateARFDataToRoot::StoreARFData(GateDigi * aDigi)
  {
  G4ThreeVector position = aDigi->GetGlobalPos();
  G4ThreeVector PosAtVertex = aDigi->GetSourcePosition();
  mNbofStoredPhotons++;
  if (mNbofStoredPhotons % 1000 == 0)
    {
    G4cout << " number of stored photons    " << mNbofStoredPhotons << Gateendl;
    G4cout << " number of NbOfSourcePhotons " << mNbOfSourcePhotons << Gateendl;
    }
  /*compute projection of the energy deposition location onto the projection plane as the intersection of the plane X=m_X
   and the line passing through totalposition with unit vector director Indirection, the incident direction */
  if (mDrfProjectionMode == 0) /*smooth positions by substracting source emission position */
    {
    mArfData.mProjectionPositionX = position.z() - PosAtVertex.z();
    mArfData.mProjectionPositionY = position.y() - PosAtVertex.y();
    }
  else if (mDrfProjectionMode == 1) /* line project */
    {
    G4ThreeVector Indirection = position - PosAtVertex;
    G4double magnitude = Indirection.mag();
    Indirection /= magnitude;
    G4double t = (mXPlane - position.x()) / Indirection.x();
    mArfData.mProjectionPositionX = position.z() + t * Indirection.z();
    mArfData.mProjectionPositionY = position.y() + t * Indirection.y();
    }
  else if (mDrfProjectionMode == 2) /* orthogonal projection */
    {
    mArfData.mProjectionPositionX = position.z();
    mArfData.mProjectionPositionY = position.y();
    }
  mArfData.mDepositedEnergy = aDigi->GetEnergy();
  mArfDataTree->Fill();
  return 1;

  }

void GateARFDataToRoot::DisplayARFStatistics()
  {
  G4cout << " Source Photons Statistics For ARF Data " << Gateendl;
  G4cout << " Camera heads number                             " << mNbOfHeads<< Gateendl;
  G4cout << " Source Photons                                  " << mNbOfSourcePhotons<< Gateendl;
  G4cout << " Detected Photons                                " << mNbofStoredPhotons << Gateendl;
  G4cout << " Source Photons Going Outside the Camera         " << mOutCamera<< Gateendl;
  G4cout << " Source Photons Going Inside the Camera          " << mInCamera << Gateendl;
  G4cout << " Source Photons Going Outside the Crystal        " << mNbofGoingOutPhotons<< Gateendl;
  G4cout << " Source Photons Going Inside the Crystal         " << mNbofGoingInPhotons<< Gateendl;
  G4cout << " Source Photons Killed Inside the Collimator     " << mNbofKilledInsideColliPhotons<< Gateendl;
  G4cout << " Source Photons Killed Inside the Camera         " << mNbofKilledInsideCamera<< Gateendl;
  G4cout << " Source Photons Killed Inside the Crystal        " << mNbofKilledInsideCrystalPhotons<< Gateendl;
  }

void GateARFDataToRoot::RecordStep(const G4Step*)
  {
  }

void GateARFDataToRoot::RecordVoxels(GateVGeometryVoxelStore*)
  {
  }

#endif

