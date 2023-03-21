/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#ifndef GateARFDataToRoot_H
#define GateARFDataToRoot_H
#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
#include "GateVOutputModule.hh"

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"

#include "globals.hh"
#include <fstream>

#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"

class GateARFDataToRootMessenger;
class GateDigi;
class GateSteppingAction;

class GateARFData
  {
public:
  G4double mDepositedEnergy; /* deposited energy */
  G4double mProjectionPositionY;
  G4double mProjectionPositionX; /* Projection position on the detection plane */
  };

class GateARFDataToRoot: public GateVOutputModule
  {
public:
  GateARFDataToRoot(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode);
  virtual ~GateARFDataToRoot();
  const G4String& GiveNameOfFile();
  void RecordBeginOfAcquisition();
  void RecordEndOfAcquisition();
  void RecordBeginOfRun(const G4Run *);
  void RecordEndOfRun(const G4Run *);
  void RecordBeginOfEvent(const G4Event *);
  void RecordEndOfEvent(const G4Event *);
  void RecordDigitizer(const G4Event *);
  void RecordStep(const G4Step*);
  void RecordVoxels(GateVGeometryVoxelStore*);
  void RecordStepWithVolume(const GateVVolume*, const G4Step*)
    {
    }
  void RecordTracks(GateSteppingAction*)
    {
    }
  void RegisterNewSingleDigiCollection(const G4String& aCollectionName, G4bool outputFlag);
  void RegisterNewCoincidenceDigiCollection(const G4String&, G4bool)
    {
    }

  void SetVerboseLevel(G4int val)
    {
    GateVOutputModule::SetVerboseLevel(val);

    }
  G4int StoreARFData(GateDigi*);
  void SetProjectionPlane(G4double aX)
    {
    mXPlane = aX;
    }

  G4ThreeVector GetPositionAtVertex();
  void SetPositionAtVertex(G4ThreeVector);
  G4ThreeVector GetVertexMomentumDirection();
  void SetVertexMomentumDirection(G4ThreeVector);

  /*! Implementation of the pure virtual method ProcessHits(). */
  /*! This methods generates a GateHit and stores it into the SD's hit collection */

  void CloseARFDataRootFile();

  void SetARFDataRootFileName(G4String);

  void IncrementNbOfSourcePhotons();

  long unsigned int GetNbOfGoingOutPhotons()
    {
    return mNbofGoingOutPhotons;
    }
  long unsigned int GetNbOfInPhotons()
    {
    return mNbofGoingInPhotons;
    }
  void IncrementGoingInPhotons()
    {
    mNbofGoingInPhotons++;
    }
  void IncrementGoingOutPhotons()
    {
    mNbofGoingOutPhotons++;
    }
  void IncrementKilledInsideCrystalPhotons()
    {
    mNbofKilledInsideCrystalPhotons++;
    }
  void IncrementKilledInsideColliPhotons()
    {
    mNbofKilledInsideColliPhotons++;
    }
  void IncrementKilledInsideCamera()
    {
    mNbofKilledInsideCamera++;
    }
  void IncrementInCamera()
    {
    mInCamera++;
    }
  void IncrementOutCamera()
    {
    mOutCamera++;
    }
  void DisplayARFStatistics();
  G4int IsCounted()
    {
    return mIsCounted;
    }
  G4int IsCountedOut()
    {
    return mIsCountedOut;
    }
  void SetCounted()
    {
    mIsCounted = 1;
    }
  void SetCountedOut()
    {
    mIsCountedOut = 1;
    }
  void SetNHeads(G4int N)
    {
    mNbOfHeads = N;
    }
  void setDRFDataprojectionmode(G4int opt)
    {
    mDrfProjectionMode = opt;
    }

private:
  GateARFDataToRootMessenger* mRootMessenger;
  G4String mArfDataFilename; /* the naeof the root output file */
  TFile* mArfDataFile; /* the root file */
  TTree* mArfDataTree; /* the root tree */
  TTree* mNbOfPhotonsTree;
  /* the datas to be saved in the root file */
  GateARFData mArfData;
  G4int mDrfProjectionMode;
  G4String mSingleDigiCollectionName; /* the singledigi collection name */
  G4RotationMatrix mRotationMatrix;
  G4ThreeVector mTranslation;
  G4double mXPlane; /* this is the YZ projection plane where we project energy deposition coordinates */
  ULong64_t mInCamera;
  ULong64_t mOutCamera;
  ULong64_t mNbOfSimuPhotons;
  ULong64_t mNbOfSourcePhotons;
  ULong64_t mNbofGoingOutPhotons;
  ULong64_t mNbofStraightPhotons;
  ULong64_t mNbofGoingInPhotons;
  ULong64_t mNbofKilledInsideCrystalPhotons;
  ULong64_t mNbofKilledInsideColliPhotons;
  ULong64_t mNbofKilledInsideCamera;
  ULong64_t mNbofBornInsidePhotons;
  ULong64_t mNbofStoredPhotons;
  G4int mIsCounted;
  G4int mIsCountedOut; /* flag to know the in going photon has been counted or not */
  G4int mHeadID;
  G4int mNbOfHeads;
  };

#endif
#endif
