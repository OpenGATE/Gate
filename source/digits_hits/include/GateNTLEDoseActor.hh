/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*!
  \class  GateNTLEDoseActor
  \authors: Halima Elazhar (halima.elazhar@ihpc.cnrs.fr)
            Thomas Deschler (thomas.deschler@iphc.cnrs.fr)
*/

#ifndef GATENTLEDOSEACTOR_HH
#define GATENTLEDOSEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateNTLEDoseActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateKermaFactorHandler.hh"

#include <G4UnitsTable.hh>

#include <TMultiGraph.h>

class GateNTLEDoseActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateNTLEDoseActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateNTLEDoseActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableEdepImage            (bool b) { mIsEdepImageEnabled              = b; }
  void EnableEdepSquaredImage     (bool b) { mIsEdepSquaredImageEnabled       = b; }
  void EnableEdepUncertaintyImage (bool b) { mIsEdepUncertaintyImageEnabled   = b; }

  void EnableDoseImage            (bool b) { mIsDoseImageEnabled              = b; }
  void EnableDoseSquaredImage     (bool b) { mIsDoseSquaredImageEnabled       = b; }
  void EnableDoseUncertaintyImage (bool b) { mIsDoseUncertaintyImageEnabled   = b; }

  void EnableFluxImage            (bool b) { mIsFluxImageEnabled              = b; }
  void EnableFluxSquaredImage     (bool b) { mIsFluxSquaredImageEnabled       = b; }
  void EnableFluxUncertaintyImage (bool b) { mIsFluxUncertaintyImageEnabled   = b; }

  void EnableDoseCorrection       (bool b) { mIsDoseCorrectionEnabled         = b; }
  void EnableDoseCorrectionTLE    (bool b) { mIsDoseCorrectionTLEEnabled      = b; }

  void EnableKFExtrapolation      (bool b) { mIsKFExtrapolated                = b; }
  void EnableKFDA                 (bool b) { mIsKFDA                          = b; }
  void EnableKermaFactorDump      (bool b) { mIsKermaFactorDumped             = b; }
  void EnableKillSecondary        (bool b) { mIsKillSecondaryEnabled          = b; }

  void EnableKermaEquivalentFactor      (bool b) { mIsKermaEquivalentFactorEnabled       = b; }
  void EnablePhotonKermaEquivalentFactor(bool b) { mIsPhotonKermaEquivalentFactorEnabled = b; }

  virtual void BeginOfRunAction  (const G4Run*);
  virtual void BeginOfEventAction(const G4Event*);

  virtual void UserSteppingAction(const GateVVolume*, const G4Step*);
  virtual void UserSteppingActionInVoxel(const int  , const G4Step*);
  virtual void UserPreTrackActionInVoxel(const int  , const G4Track*) {}
  virtual void UserPostTrackActionInVoxel(const int , const G4Track*) {}

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateNTLEDoseActor(G4String name, G4int depth=0);
  GateNTLEDoseActorMessenger * pMessenger;

  GateImageWithStatistic mEdepImage;
  GateImageWithStatistic mDoseImage;
  GateImageWithStatistic mFluxImage;

  GateImage mLastHitEventImage;

  GateKermaFactorHandler* mKFHandler;

  G4String mEdepFilename;
  G4String mDoseFilename;
  G4String mFluxFilename;

  bool mIsLastHitEventImageEnabled;

  bool mIsEdepImageEnabled;
  bool mIsEdepSquaredImageEnabled;
  bool mIsEdepUncertaintyImageEnabled;

  bool mIsDoseImageEnabled;
  bool mIsDoseSquaredImageEnabled;
  bool mIsDoseUncertaintyImageEnabled;

  bool mIsFluxImageEnabled;
  bool mIsFluxSquaredImageEnabled;
  bool mIsFluxUncertaintyImageEnabled;

  bool mIsDoseCorrectionEnabled;
  bool mIsDoseCorrectionTLEEnabled;

  bool mIsKFExtrapolated;
  bool mIsKFDA;
  bool mIsKermaFactorDumped;
  bool mIsKillSecondaryEnabled;

  bool mIsKermaEquivalentFactorEnabled;
  bool mIsPhotonKermaEquivalentFactorEnabled;

  int mCurrentEvent;

  std::vector<G4String> mMaterialList;

  TMultiGraph* mg;
};

MAKE_AUTO_CREATOR_ACTOR(NTLEDoseActor,GateNTLEDoseActor)

#endif
