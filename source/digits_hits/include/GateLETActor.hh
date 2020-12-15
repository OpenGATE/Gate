/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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

  void SetLETtoWater(bool b) { mIsLETtoWaterEnabled = b; }
  void SetParallelCalculation(bool b) { mIsParallelCalculationEnabled = b; }
  void SetLETType(G4String s) { mAveragingType = s; }
  void SetMaterial(G4String s) { mSetMaterial = s; }
  void SetCutVal(G4double d) { mCutVal = d; }
  void SetLETthrMin(G4double d) { mLETthrMin = d; }
  void SetLETthrMax(G4double d) { mLETthrMax = d; }

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
  
  //virtual void polynomial(double* coefs, double deg, double x) {double yv;}
  virtual double polynomial(double * coefs, int deg, double x);

  int mCurrentEvent;
  bool mIsLETtoWaterEnabled;
  G4String mAveragingType;
  G4String mSetMaterial;
  
  GateImageDouble mWeightedLETImage;
  GateImageDouble mNormalizationLETImage;
  GateImageDouble mDoseTrackAverageLETImage;
  G4String mLETFilename;
  G4String numeratorFileName;
  G4String denominatorFileName;
  G4String sigmaFilename;
  
  bool mKGrosswendt;
  G4double mCutVal;
  G4double mLETthrMin;
  G4double mLETthrMax;
  
  
  double k_FitParWAir;
  bool mIsSwairApprox;
  bool mIsMeanEnergyToProduceIonPairInAir;
  bool mIsMeanEnergyToProduceIonPairInAirAR;

  bool mIsDoseAveraged;
  bool mIsTrackAveraged;

  bool mIsTrackAverageDEDX;
  bool mIsTrackAverageEdepDX;

  bool mIsDoseAverageDEDX;
  bool mIsDoseAverageEdepDX;
  
  bool mIsAverageKinEnergy;
  
  bool mIsGqq0EBT31stOrder;
  bool mIsGqq0EBT34thOrder;


  bool mIsParallelCalculationEnabled;

  G4EmCalculator * emcalc;
  
  StepHitType mUserStepHitType;
};

MAKE_AUTO_CREATOR_ACTOR(LETActor,GateLETActor)

#endif /* end #define GATELETACTOR_HH */
