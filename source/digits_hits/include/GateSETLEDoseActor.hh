/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*!
  \class  GateSETLEDoseActor
  \author fabien.baldacci@creatis.insa-lyon.fr
	  francois.smekens@creatis.insa-lyon.fr
  
 */

#ifndef GATESETLEDOSEACTOR_HH
#define GATESETLEDOSEACTOR_HH

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateSETLEDoseActorMessenger.hh"
#include "GateSETLEMultiplicityActor.hh"
#include "GateImageWithStatistic.hh"
#include "GateMaterialMuHandler.hh"
#include "G4SteppingManager.hh"

class GateSETLEDoseActor : public GateVImageActor
{
 public: 
  
  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateSETLEDoseActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateSETLEDoseActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableDoseImage(bool b) { mIsDoseImageEnabled = b; }
  void EnableDoseUncertaintyImage(bool b) { mIsDoseUncertaintyImageEnabled = b; }
  void EnablePrimaryDoseImage(bool b) { mIsPrimaryDoseImageEnabled = b; }
  void EnablePrimaryDoseUncertaintyImage(bool b) { mIsPrimaryDoseUncertaintyImageEnabled = b; }
  void EnableSecondaryDoseImage(bool b) { mIsSecondaryDoseImageEnabled = b; }
  void EnableSecondaryDoseUncertaintyImage(bool b) { mIsSecondaryDoseUncertaintyImageEnabled = b; }
  void EnableHybridino(bool b) { mIsHybridinoEnabled = b; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);
  
  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track* t);
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track* t);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  void SetPrimaryMultiplicity(int m) { mPrimaryMultiplicity = m; }
  void SetSecondaryMultiplicity(int m) { mSecondaryMultiplicity = m; }
  void SetSecondaryMultiplicity(double t, double n) { mSecondaryMultiplicity = int( ((t / n) - 26.6557E-4) / 1.26961E-4 ); }
  
  int GetPrimaryMultiplicity() { return mPrimaryMultiplicity; }
  int GetSecondaryMultiplicity() { return mSecondaryMultiplicity; }

  void InitializeMaterialAndMuTable();
  bool IntersectionBox(G4ThreeVector, G4ThreeVector);
  double RayCast(bool, double, double, G4ThreeVector, G4ThreeVector);
 /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void clear(){ResetData();}
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}
  
protected:
  GateSETLEDoseActor(G4String name, G4int depth=0);
  GateSETLEDoseActorMessenger *pMessenger;
  
  GateImageWithStatistic mDoseImage;
  GateImage mLastHitEventImage;
  bool mIsLastHitEventImageEnabled;
  bool mIsDoseImageEnabled;
  bool mIsDoseUncertaintyImageEnabled;
  
  GateImageWithStatistic mPrimaryDoseImage;
  GateImage mPrimaryLastHitEventImage;
  bool mIsPrimaryLastHitEventImageEnabled;
  bool mIsPrimaryDoseImageEnabled;
  bool mIsPrimaryDoseUncertaintyImageEnabled;
  
  GateImageWithStatistic mSecondaryDoseImage;
  GateImage mSecondaryLastHitEventImage;
  bool mIsSecondaryLastHitEventImageEnabled;
  bool mIsSecondaryDoseImageEnabled;
  bool mIsSecondaryDoseUncertaintyImageEnabled;
  
  GateMaterialMuHandler* mMaterialHandler;
  G4String mDoseFilename;
  G4String mPrimaryDoseFilename;
  G4String mSecondaryDoseFilename;

  G4double ConversionFactor;
  G4double VoxelVolume;
  const G4MaterialCutsCouple *mWorldCouple;
  
  GateSETLEMultiplicityActor *pSETLEMultiplicityActor;
  int mPrimaryMultiplicity;
  int mSecondaryMultiplicity;
  bool mIsHybridinoEnabled;
  std::vector<RaycastingStruct> *mListOfRaycasting;

  bool mIsMuTableInitialized;
  std::vector<GateMuTable *> mListOfMuTable;
  
  int mCurrentEvent;
  G4SteppingManager *mSteppingManager;
  G4RotationMatrix mRotationMatrix;
  G4AffineTransform worldToVolume;

  // raycasting members
  double mBoxMin[3];
  double mBoxMax[3];
  double mRayOrigin[3];
  double mRayDirection[3];
  double mNearestDistance;
  double mFarthestDistance;
  double mTotalLength;
  int mLineSize;
  int mPlaneSize;
};

MAKE_AUTO_CREATOR_ACTOR(SETLEDoseActor,GateSETLEDoseActor)

#endif /* end #define GATEHYBRIDDOSEACTOR_HH */
