/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateCylindricalEdepActor
  \author A.Resch
  based on GateDoseActor
  \date	March 2011

 */


#ifndef GATECYLINDRICALEDEPACTOR_HH
#define GATECYLINDRICALEDEPACTOR_HH

#include <G4NistManager.hh>

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateCylindricalEdepActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateVoxelizedMass.hh"
#include "G4VProcess.hh"

class G4EmCalculator;

class GateCylindricalEdepActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateCylindricalEdepActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateCylindricalEdepActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableEdepImage(bool b) { mIsEdepImageEnabled = b; }
  void EnableFluenceImage(bool b) { mIsFluenceImageEnabled = b; }
  
 
  void EnableDoseImage(bool b) { mIsDoseImageEnabled = b; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track);
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  //  Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Scorer related
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

protected:
  GateCylindricalEdepActor(G4String name, G4int depth=0);
  GateCylindricalEdepActorMessenger* pMessenger;
  GateVoxelizedMass mVoxelizedMass;

  int mCurrentEvent;
  StepHitType mUserStepHitType;
   bool mIsCylindricalSymmetryImage;
  bool mIsEdepImageEnabled;
  bool mIsFluenceImageEnabled;
  bool mIsDoseImageEnabled;

  GateImageWithStatistic mEdepImage;
  GateImageWithStatistic mFluenceImage;
  
  GateImageWithStatistic mDoseImage;
  
  G4String mEdepFilename;
  G4String mFluenceFilename;
  G4String mDoseFilename;
};

MAKE_AUTO_CREATOR_ACTOR(CylindricalEdepActor,GateCylindricalEdepActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
