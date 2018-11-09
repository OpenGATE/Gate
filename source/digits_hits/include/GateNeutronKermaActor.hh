/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateNeutronKermaActor
  \author Thomas DESCHLER (thomas@deschler.fr)
  \date	May 2017
 */

#ifndef GATENEUTRONKERMAACTOR_HH
#define GATENEUTRONKERMAACTOR_HH

#include <G4NistManager.hh>

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateNeutronKermaActorMessenger.hh"
#include "GateImageWithStatistic.hh"

class G4EmCalculator;

class GateNeutronKermaActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateNeutronKermaActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateNeutronKermaActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableEdepImage            (bool b) { mIsEdepImageEnabled            = b; }
  void EnableEdepSquaredImage     (bool b) { mIsEdepSquaredImageEnabled     = b; }
  void EnableEdepUncertaintyImage (bool b) { mIsEdepUncertaintyImageEnabled = b; }

  void EnableDoseImage            (bool b) { mIsDoseImageEnabled            = b; }
  void EnableDoseSquaredImage     (bool b) { mIsDoseSquaredImageEnabled     = b; }
  void EnableDoseUncertaintyImage (bool b) { mIsDoseUncertaintyImageEnabled = b; }

  void EnableNumberOfHitsImage(bool b) { mIsNumberOfHitsImageEnabled = b; }
  void EnableDoseNormalisation(bool b) { mIsDoseNormalisationEnabled = b; mDoseImage.SetScaleFactor(1.0); }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserSteppingActionInVoxel (const int, const G4Step*);
  virtual void UserPreTrackActionInVoxel (const int, const G4Track*) {}
  virtual void UserPostTrackActionInVoxel(const int, const G4Track*) {}

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateNeutronKermaActor(G4String name, G4int depth=0);
  GateNeutronKermaActorMessenger * pMessenger;

  int mCurrentEvent;

  bool mIsLastHitEventImageEnabled;

  bool mIsEdepImageEnabled;
  bool mIsEdepSquaredImageEnabled;
  bool mIsEdepUncertaintyImageEnabled;

  bool mIsDoseImageEnabled;
  bool mIsDoseSquaredImageEnabled;
  bool mIsDoseUncertaintyImageEnabled;

  bool mIsNumberOfHitsImageEnabled;
  bool mIsDoseNormalisationEnabled;
  bool mIsDoseToWaterNormalisationEnabled;

  GateImageWithStatistic mEdepImage;
  GateImageWithStatistic mDoseImage;

  GateImage mNumberOfHitsImage;
  GateImage mLastHitEventImage;

  G4String mEdepFilename;
  G4String mDoseFilename;
  G4String mNbOfHitsFilename;

  G4EmCalculator * emcalc;
};

MAKE_AUTO_CREATOR_ACTOR(NeutronKermaActor,GateNeutronKermaActor)

#endif
