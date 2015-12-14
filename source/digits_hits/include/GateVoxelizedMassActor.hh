/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateVoxelizedMassActor
  \author Thomas DESCHLER (thomas.deschler@iphc.cnrs.fr)
  \date	October 2015
*/

#ifndef GATEVOXELIZEDMASSACTOR_HH
#define GATEVOXELIZEDMASSACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateVoxelizedMassActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateVoxelizedMass.hh"

class GateVoxelizedMassActor : public GateVImageActor
{
 public:

  virtual ~GateVoxelizedMassActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateVoxelizedMassActor)

  virtual void Construct();

  void EnableMassImage(bool b) {mIsMassImageEnabled = b;}

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * /*event*/) {}

  virtual void UserSteppingActionInVoxel(const int /*index*/, const G4Step* /*step*/) {}
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*track*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  //  Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Scorer related
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:

  GateVoxelizedMassActor(G4String name, G4int depth=0);
  GateVoxelizedMassActorMessenger* pMessenger;
  GateVoxelizedMass pVoxelizedMass;

  int mCurrentEvent;
  StepHitType mUserStepHitType;

  bool mIsMassImageEnabled;
  GateImageDouble mMassImage;// ATTENTION : mDoseImage.GetValueImage()=GateImageDouble!!!
  G4String mMassFilename;

  std::vector<double> voxelMass;

};

MAKE_AUTO_CREATOR_ACTOR(VoxelizedMassActor,GateVoxelizedMassActor)

#endif   
