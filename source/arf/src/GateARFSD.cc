/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFSD.hh"
#include "GateHit.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ios.hh"
#include "G4VProcess.hh"
#include "G4TransportationManager.hh"
#include "GateVSystem.hh"
#include "GateRotationMove.hh"
#include "GateOrbitingMove.hh"
#include "GateEccentRotMove.hh"
#include "GateSystemListManager.hh"
#include "GateVVolume.hh"
#include "GateDigitizer.hh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include "globals.hh"
#include "GateARFSDMessenger.hh"
#include "GateARFTableMgr.hh" /* manages the ARF tables */
#include "GateVVolume.hh"
#include "G4ThreeVector.hh"
#include <ctime>
#include "GateBox.hh"
#include "GateToProjectionSet.hh"
#include "GateOutputMgr.hh"
#include "TH1D.h"

/* Name of the hit collection */
const G4String GateARFSD::mArfHitCollectionName = "ARFCollection";

/* Constructor */
GateARFSD::GateARFSD(const G4String& pathname, const G4String & name) :
    G4VSensitiveDetector(pathname), mSystem(0), mName(name)
  {
  collectionName.insert(mArfHitCollectionName);
  mNbOfRejectedPhotons = 0;
  mMessenger = new GateARFSDMessenger(this);
  mInserter = 0;
  mProjectionSet = 0;
  mArfTableMgr = new GateARFTableMgr(GetName(), this);
  mFile = 0;
  mSinglesTree = 0;
  mNbOfSimuPhotons = 0;
  mNbofGoingOutPhotons = 0;
  mNbofStraightPhotons = 0;
  mNbofGoingInPhotons = 0;
  mNbOfSourcePhotons = 0;
  mNbOfGoodPhotons = 0;
  mNbofStoredPhotons = 0;
  mNbOfHeads = 0;
  mEnergyDepositionThreshold = 0.;
  mHeadID = -1;
  mDetectorXDepth = 0.;
  mArfStage = -2;
  mShortcutARF = false;
  GateMessage("Geometry", 2, "ARF Sensitive Detector created");

  }

/* Destructor */
GateARFSD::~GateARFSD()
  {
  delete mMessenger;
  delete mArfTableMgr;
  }

GateARFSD *GateARFSD::Clone() const
  {
  auto clone = new GateARFSD(SensitiveDetectorName, mName);
  clone->thePathName = thePathName;
  clone->fullPathName = fullPathName;
  clone->verboseLevel = verboseLevel;
  clone->active = active;
  clone->ROgeometry = ROgeometry;
  clone->filter = filter;

  clone->mSystem = mSystem;
  clone->mArfHitCollection = mArfHitCollection;
  clone->mInserter = mInserter;
  clone->mFile = mFile;
  clone->mSinglesTree = mSinglesTree;
  clone->mNbOfPhotonsTree = mNbOfPhotonsTree;
  clone->mNbOfSourcePhotons = mNbOfSourcePhotons;
  clone->mNbOfSimuPhotons = mNbOfSimuPhotons;
  clone->mNbofGoingOutPhotons = mNbofGoingOutPhotons;
  clone->mNbofGoingInPhotons = mNbofGoingInPhotons;
  clone->mNbofStraightPhotons = mNbofStraightPhotons;
  clone->mNbofStoredPhotons = mNbofStoredPhotons;
  clone->mNbOfGoodPhotons = mNbOfGoodPhotons;
  clone->mInCamera = mInCamera;
  clone->mOutCamera = mOutCamera;
  clone->mNbOfRejectedPhotons = mNbOfRejectedPhotons;
  clone->mArfData = mArfData;
  clone->mProjectionSet = mProjectionSet;
  clone->mHeadID = mHeadID;
  clone->mNbOfHeads = mNbOfHeads;
  clone->mDetectorXDepth = mDetectorXDepth;
  clone->mEnergyWindows = mEnergyWindows;
  clone->mEnergyWindowsNumberOfPrimaries = mEnergyWindowsNumberOfPrimaries;
  clone->mEnergyDepositionThreshold = mEnergyDepositionThreshold;
  clone->mArfStage = mArfStage;
  clone->mShortcutARF = mShortcutARF;

  return clone;
  }

/* Method overloading the virtual method Initialize() of G4VSensitiveDetector */
void GateARFSD::Initialize(G4HCofThisEvent*HCE)
  {
  static int HCID = -1; /* Static variable storing the hit collection ID */
  /* Not thread safe but moving to local variable doesn't work */
  /* Creation of a new hit collection */
  mArfHitCollection = new GateHitsCollection(SensitiveDetectorName, mArfHitCollectionName);
  /* We store the hit collection ID into the static variable HCID */
  if (HCID < 0)
    {
    HCID = GetCollectionID(0);
    }
  /* Add the hit collection to the G4HCofThisEvent */
  HCE->AddHitsCollection(HCID, mArfHitCollection);

  }

/* Implementation of the pure virtual method ProcessHits().
 This methods generates a GateGeomColliHit and stores it into the SD's hit collection */

G4bool GateARFSD::ProcessHits(G4Step*step, G4TouchableHistory*)
  {
  if (!mShortcutARF)
    {
    G4Track* track = static_cast<G4Track*>(step->GetTrack());
    /* TODO Check if necessary */
    /*if (track->GetParentID() != 0)
     {
     return false;
     }
     */
    track->SetTrackStatus(fKillTrackAndSecondaries);
    /* Get the step-points */
    G4StepPoint *preStepPoint = step->GetPreStepPoint();
    G4StepPoint *postStepPoint = step->GetPostStepPoint();
    const G4VProcess*processDefinedStep = postStepPoint->GetProcessDefinedStep();

    /*  For all processes except transportation, we select the PostStepPoint volume
     For the transportation, we select the PreStepPoint volume */
    const G4TouchableHistory* touchable;
    if (processDefinedStep->GetProcessType() == fTransportation)
      {
      touchable = (const G4TouchableHistory*) (preStepPoint->GetTouchable());
      }
    else
      {
      touchable = (const G4TouchableHistory*) (postStepPoint->GetTouchable());
      }
    GateVolumeID volumeID(touchable);
    mHeadID = volumeID.GetVolume(volumeID.GetCreatorDepth("SPECThead"))->GetCopyNo();
    if (volumeID.IsInvalid())
      {
      G4Exception("GateARFSD::ProcessHits",
                  "ProcessHits",
                  FatalException,
                  "Could not get the volume ID! Aborting!");
      }

    /* Now we compute the position in the current frame to be able to extract the angles theta and phi */
    G4ThreeVector localPosition = volumeID.MoveToBottomVolumeFrame(track->GetPosition());
    G4ThreeVector vertexPosition = volumeID.MoveToBottomVolumeFrame(step->GetPreStepPoint()->GetPosition());
    G4ThreeVector direction = localPosition - vertexPosition;

    G4double magnitude = direction.mag();
    direction /= magnitude;
    ComputeProjectionSet(localPosition,
                         direction,
                         track->GetTotalEnergy(),
                         preStepPoint->GetWeight());
    }
  return true;
  }

G4int GateARFSD::PrepareCreatorAttachment(GateVVolume* creator)
  {
  GateVSystem* creatorSystem = GateSystemListManager::GetInstance()->FindSystemOfCreator(creator);
  if (!creatorSystem)
    {
    G4cout << Gateendl<< Gateendl << "[GateARFSD::PrepareCreatorAttachment]:\n"
    << "Volume '" << creator->GetObjectName() << "' does not belong to any system.\n"
    << "Your volume must belong to a system to be used with a GeomColliSD.\n"
    << "Attachment request ignored --> you won't have any hit output from this volume!!!\n";
    return -1;
    }
  if (mSystem)
    {
    if (creatorSystem != mSystem)
      {
      G4cout << Gateendl<< Gateendl << "[GateARFSD::PrepareCreatorAttachment]:\n"
      << "Volume '" << creator->GetObjectName() << "' belongs to system '" << creatorSystem->GetObjectName() << "'\n"
      << "while the GeomColliSD has already been attached to a volume from another system ('" << mSystem->GetObjectName()<< "').\n"
      << "Attachment request ignored --> you won't have any hit output from this volume!!!\n";
      return -1;
      }
    }
  else
    {
    SetSystem(creatorSystem);}
  return 0;
  }

/* Set the system to which the SD is attached */
void GateARFSD::SetSystem(GateVSystem* system)
  {
  mSystem = system;
  GateDigitizer::GetInstance()->SetSystem(system);
  }

void GateARFSD::computeTables()
  { /* open the root files generated from the ARF simu */

  if (mArfStage != 1)
    {
    G4Exception("GateARFSD::computeTable",
                "computeTable",
                FatalException,
                "Illegal state of the Gate ARF Sensitive Detector");
    }

  G4cout << "GateARFSD::computeTables() -  Computing ARF Tables for Sensitive Detector "
         << GetName()
         << Gateendl;

  time_t timeBefore = time(NULL);
  if (mArfTableMgr->InitializeTables() == 1)
    {
    return;
    }
  G4double* nbSourcePhotons = new G4double[mEnergyWindows.size()];
  G4int tableIndex = 0;
  G4int totalNumberOfSingles = 0;
  G4String rootName;
  ULong64_t tempNbofGoingOutPhotons = 0;
  ULong64_t tempNbofGoingInPhotons = 0;
  ULong64_t tempNbOfSourcePhotons = 0;
  ULong64_t tempNbofStoredPhotons = 0;
  ULong64_t tempInCamera = 0;
  ULong64_t tempOutCamera = 0;
  for (unsigned int numberOfWindows = 0; numberOfWindows < mEnergyWindows.size(); numberOfWindows++)
    {
    mNbOfSimuPhotons = 0;
    mNbofGoingOutPhotons = 0;
    mNbofStraightPhotons = 0;
    mNbofGoingInPhotons = 0;
    mNbOfSourcePhotons = 0;
    mNbofStoredPhotons = 0;
    mInCamera = 0;
    mOutCamera = 0;
    rootName = mEnergyWindows[numberOfWindows] + ".root";
    totalNumberOfSingles = 0;
    tempNbofGoingOutPhotons = 0;
    tempNbofGoingInPhotons = 0;
    tempNbOfSourcePhotons = 0;
    tempNbofStoredPhotons = 0;
    tempInCamera = 0;
    tempOutCamera = 0;

    for (G4int i = 0; i < mEnergyWindowsNumberOfPrimaries[numberOfWindows]; i++)
      {
      if (mEnergyWindowsNumberOfPrimaries[numberOfWindows] > 1)
        {
        std::stringstream s;
        s << i;
        if (mEnergyWindowsNumberOfPrimaries[numberOfWindows] <= 10)
          {
          rootName = mEnergyWindows[numberOfWindows] + "_" + s.str() + ".root";
          }
        else if (mEnergyWindowsNumberOfPrimaries[numberOfWindows] <= 100)
          {
          if (i < 10)
            {
            rootName = mEnergyWindows[numberOfWindows] + "_0" + s.str() + ".root";
            }
          else
            {
            rootName = mEnergyWindows[numberOfWindows] + "_" + s.str() + ".root";
            }
          }
        }
      if (mFile != 0)
        {
        delete mFile;
        mFile = 0;
        }
      mFile = new TFile(rootName.c_str(), "READ", "ROOT filefor ARF purpose");
      G4cout << "GateARFSD::computeTables():::::: Reading ROOT File  " << rootName << Gateendl;
      mSinglesTree = (TTree*) (mFile->Get("theTree"));
      G4cout << " m_singlesTree = " << mSinglesTree << Gateendl;
      mSinglesTree->SetBranchAddress("Edep", &mArfData.mDepositedEnergy);
      mSinglesTree->SetBranchAddress("outY", &mArfData.mProjectionPositionY);
      mSinglesTree->SetBranchAddress("outX", &mArfData.mProjectionPositionX);
      mNbOfPhotonsTree = (TTree*) (mFile->Get("theNumberOfPhoton"));
      G4cout << " m_NbOfPhotonsTree = " << mNbOfPhotonsTree << Gateendl;
      mNbOfPhotonsTree->SetBranchAddress("NOfOutGoingPhot", &tempNbofGoingOutPhotons);
      mNbOfPhotonsTree->SetBranchAddress("NbOfInGoingPhot", &tempNbofGoingInPhotons);
      mNbOfPhotonsTree->SetBranchAddress("NbOfSourcePhot", &tempNbOfSourcePhotons);
      mNbOfPhotonsTree->SetBranchAddress("NbOfStoredPhotons", &tempNbofStoredPhotons);
      mNbOfPhotonsTree->SetBranchAddress("NbOfInCameraPhot", &tempInCamera);
      mNbOfPhotonsTree->SetBranchAddress("NbOfOutCameraPhot", &tempOutCamera);
      mNbOfPhotonsTree->SetBranchAddress("NbOfHeads", &mNbOfHeads);
      mNbOfPhotonsTree->GetEntry(0);
      mNbofGoingOutPhotons += tempNbofGoingOutPhotons;
      mNbofGoingInPhotons += tempNbofGoingInPhotons;
      mNbOfSourcePhotons += tempNbOfSourcePhotons;
      mNbofStoredPhotons += tempNbofStoredPhotons;
      mInCamera += tempInCamera;
      mOutCamera += tempOutCamera;
      G4cout << " In File " << rootName << Gateendl;
      G4cout << " Total number of Source photons Going Out Crystal  "
             << tempNbofGoingOutPhotons
             << Gateendl;
      G4cout << " Total number of Source photons Going In Crystal   "
             << tempNbofGoingInPhotons
             << Gateendl;
      G4cout << " Total number of Source photons                    "
             << tempNbOfSourcePhotons
             << Gateendl;
      G4cout << " Total number of Source photons Going In Camera    " << tempInCamera << Gateendl;
      G4cout << " Total number of Source photons Going Out Camera   " << tempOutCamera << Gateendl;
      G4cout << " Total number of Stored photons                    "
             << tempNbofStoredPhotons
             << Gateendl;
      totalNumberOfSingles = mSinglesTree->GetEntries();
      G4cout << " File " << rootName << " contains " << totalNumberOfSingles << " entries \n";
      G4cout << " Tree m_NbOfPhotonsTree "
             << rootName
             << " contains "
             << mNbOfPhotonsTree->GetEntries()
             << " entries \n";
      for (G4int j = 0; j < totalNumberOfSingles; j++)
        {
        mSinglesTree->GetEntry(j);
        /* loop through ARF tables to get the table with the suitable energy window */
        if (mArfData.mDepositedEnergy / keV - mEnergyDepositionThreshold >= 0.)
          {
          mArfTableMgr->FillDRFTable(tableIndex,
                                     mArfData.mDepositedEnergy,
                                     mArfData.mProjectionPositionX,
                                     mArfData.mProjectionPositionY);
          }
        }
      mFile->Close();
      }
    time_t timeAfter = time(NULL);
    nbSourcePhotons[tableIndex] = mNbOfSourcePhotons * mNbOfHeads;
    G4cout << " ARF Table # "
           << tableIndex
           << "  Computation Time "
           << (timeAfter - timeBefore)
           << " seconds \n";
    tableIndex++; /* now for next ARF table */
    }
  mArfTableMgr->SetNSimuPhotons(nbSourcePhotons);
  mArfTableMgr->convertDRF2ARF();
  }

void GateARFSD::ComputeProjectionSet(const G4ThreeVector & position,
                                     const G4ThreeVector & direction,
                                     const G4double & energy,
                                     const G4double & weight,
                                     bool addEmToArfCount,
                                     unsigned int newHead)
  {
  /*
   transform to the detector frame the photon position
   we compute the direction and position relative to the detector frame
   we store also the rotation matrix and the translation of the detector relative to the world frame
   a position and aDirection are computed relative to the detector frame !
   the coordinates of the intersection of the path of the photon with the back surface of the detector
   is given by
   x = deltaX/2
   y = yin + t * uy
   z = zin + t * uz
   where
   u(ux,uy,uz) is the direction vector of the photon
   (xin,yin,zin) is the starting  position of the photon when it enters the detector
   and
   t = ( deltaX - xin ) / ux
   deltaX is the projection plane of the detector on the Ox axis
   all these coordinates are relative to the detector frame where the origin of hte detector is a t the center
   */

  G4double arfValue = mArfTableMgr->ScanTables(direction.z(), direction.y(), energy);
  /* The coordinates of the intersection of the path of the photon with the back surface of the detector
   is given by
   x = deltaX/2
   y = yin + t * uy
   z = zin + t * uz
   where
   u(ux,uy,uz) is the direction vector of the photon
   (xin,yin,zin) is the starting  position of the photon when it enters the detector
   and
   t = ( deltaX/2 - xin ) / ux
   deltaX is the dimension of the detector on the Ox axis
   all these coordinates are relative to the detector frame where the origin of the detector is a t the center */

  G4double t = (position.x() - mDetectorXDepth) / direction.x();
  G4double xP = position.z() + t * direction.z();
  G4double yP = position.y() + t * direction.y();
  /* now store projection with the GateProjectionSet Module though its method GateProjectionSet::Fill */

  if (mProjectionSet == 0)
    {
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    GateToProjectionSet* projectionSet = dynamic_cast<GateToProjectionSet*>(outputMgr->GetModule("projection"));
    if (projectionSet == 0)
      {
      G4Exception("GateARFSD::ComputeProjectionSet()",
                  "ComputeProjectionSet",
                  FatalException,
                  "ERROR No Projection Set Module has been enabled. Aborting.");
      }
    mProjectionSet = projectionSet->GetProjectionSet();
    }
  mProjectionSet->FillARF(mHeadID, yP, -xP, arfValue * weight, addEmToArfCount);

  if (mShortcutARF)
    {
    mProjectionSet->FillARF(newHead, yP, -xP, arfValue * weight, false);
    }

  }

#endif
