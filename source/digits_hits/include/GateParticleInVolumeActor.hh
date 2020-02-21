/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*!
  \class  GateParticleInVolumeActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEPARTINVOLACTOR_HH
#define GATEPARTINVOLACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateImageActorMessenger.hh"

class GateParticleInVolumeActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateParticleInVolumeActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateParticleInVolumeActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableParticleInVolumeImage(bool b) { mIsParticleInVolumeImageEnabled = b; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* t);
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}


protected:
  GateParticleInVolumeActor(G4String name, G4int depth=0);
  GateImageActorMessenger * pMessenger;

  bool mIsLastHitEventImageEnabled;
  bool mIsParticleInVolumeImageEnabled;
  bool outsideTrack;

  GateImageInt mLastHitEventImage;
  GateImage mParticleInVolumeImage;

  G4String mParticleInVolumeFilename;
  G4int mCurrentEvent;
};

MAKE_AUTO_CREATOR_ACTOR(ParticleInVolumeActor,GateParticleInVolumeActor)

#endif
