/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateStoppingPowerActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
 */

#include "GateConfiguration.h"

#ifndef GATESTOPPOWACTOR_HH
#define GATESTOPPOWACTOR_HH

#include "GateVImageActor.hh"
#include "G4UnitsTable.hh"
#include "GateImageWithStatistic.hh"
#include "GateImageActorMessenger.hh"

class GateStoppingPowerActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateStoppingPowerActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateStoppingPowerActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableStopPowerImage(bool b) { mIsStopPowerImageEnabled = b; }
  void EnableStopPowerSquaredImage(bool b) { mIsStopPowerSquaredImageEnabled = b; }
  void EnableStopPowerUncertaintyImage(bool b) { mIsStopPowerUncertaintyImageEnabled = b; }
  void EnableRelStopPowerImage(bool b) { mIsRelStopPowerImageEnabled = b; }
  void EnableRelStopPowerSquaredImage(bool b) { mIsRelStopPowerSquaredImageEnabled = b; }
  void EnableRelStopPowerUncertaintyImage(bool b) { mIsRelStopPowerUncertaintyImageEnabled = b; }
  void EnableNumberOfHitsImage(bool b) { mIsNumberOfHitsImageEnabled = b; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

 /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateStoppingPowerActor(G4String name, G4int depth=0);

  GateImageActorMessenger * pMessenger;

  int mCurrentEvent;

  bool mIsLastHitEventImageEnabled;
  bool mIsStopPowerImageEnabled;
  bool mIsRelStopPowerImageEnabled;
  bool mIsNumberOfHitsImageEnabled;

  bool mIsStopPowerSquaredImageEnabled;
  bool mIsStopPowerUncertaintyImageEnabled;
  bool mIsRelStopPowerSquaredImageEnabled;
  bool mIsRelStopPowerUncertaintyImageEnabled;

  GateImageWithStatistic mStopPowerImage;
  GateImageWithStatistic mRelStopPowerImage;
  GateImage mNumberOfHitsImage;
  GateImage mLastHitEventImage;

  G4String mStopPowerFilename;
  G4String mRelStopPowerFilename;
  G4String mNbOfHitsFilename;

};

MAKE_AUTO_CREATOR_ACTOR(StoppingPowerActor,GateStoppingPowerActor)

#endif
