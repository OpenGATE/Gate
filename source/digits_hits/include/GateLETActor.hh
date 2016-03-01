/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateLETActor
  Compute LET (Linear Energy Transfer) at a voxel level.
  \date	2016
 */

#ifndef GATELETACTOR_HH
#define GATELETACTOR_HH

#include <G4NistManager.hh>

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateLETActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "G4VProcess.hh"

class G4EmCalculator;

class GateLETActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateLETActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateLETActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void SetRestrictedFlag(bool b) { mIsRestrictedFlag = b; }
  void SetDeltaRestrictedValue(G4double d) { mDeltaRestricted = d; }
  //test AResch 25 01 2016
  void EnableLETUncertaintyImage(bool b) { mIsLETUncertaintyImageEnabled = b; }
  void SetDoseToWater(bool b) { mIsDoseToWaterEnabled = b; }
  void SetParallelCalculation(bool b) { mIsParallelCalculationEnabled = b; }
  void SetLETType(G4String s) { mAveragingType = s; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);

  // Do nothing but needed because pure virtual
  virtual void UserPreTrackActionInVoxel(const int, const G4Track*) {}
  virtual void UserPostTrackActionInVoxel(const int, const G4Track*) {}

  //  Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Scorer related
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}


protected:
  GateLETActor(G4String name, G4int depth=0);
  GateLETActorMessenger * pMessenger;

  bool mIsRestrictedFlag;
  
  int mCurrentEvent;
  bool mIsLETUncertaintyImageEnabled;
  bool mIsDoseToWaterEnabled;
  G4String mAveragingType;
  G4double mDeltaRestricted;
  GateImageDouble mLETImage;
  GateImageDouble mEdepImage;
  GateImageDouble mFinalImage;
  G4String mLETFilename;
  G4String uncertaintyFilename;
  GateImageDouble mLETSecondMomentImage;
  GateImageDouble mLETUncertaintyFinalImage;
  //GateImageWithStatistic mTestImage;

  bool mIsDoseAveraged;
  bool mIsTrackAveraged;
  
  bool mIsTrackAveragedFluenceAveraged;
  bool mIsTrackAveragedFluenceTrackAveraged;
  bool mIsTrackAveragedDXAveraged;
  bool mIsTrackAveragedDXAveragedCancelled;
  
  bool mIsDoseAveragedDEDXAveraged;
  bool mIsDoseAveragedEdepDXAveraged;
  
  GateImageDouble mLETTempImage;
  GateImageInt mNumberOfHitsImage;
  GateImageInt mLastHitEventImage;
  bool mIsRelUncertEnabled;
  
  GateImageDouble mRelUncertImage;
  
  bool mIsParallelCalculationEnabled;
  
  G4EmCalculator * emcalc;
};

MAKE_AUTO_CREATOR_ACTOR(LETActor,GateLETActor)

#endif /* end #define GATELETACTOR_HH */
