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
  
  void EnableEdepHadElasticImage(bool b) { mIsEdepHadElasticImageEnabled = b; }
  void EnableEdepInelasticImage(bool b) { mIsEdepInelasticImageEnabled = b; }
  void EnableEdepRestImage(bool b) { mIsEdepRestImageEnabled = b; }
  
  void EnableEdepSquaredImage(bool b) { mIsEdepSquaredImageEnabled = b; }
  void EnableEdepUncertaintyImage(bool b) { mIsEdepUncertaintyImageEnabled = b; }
  void EnableDoseImage(bool b) { mIsDoseImageEnabled = b; }
  void EnableDoseSquaredImage(bool b) { mIsDoseSquaredImageEnabled = b; }
  void EnableDoseUncertaintyImage(bool b) { mIsDoseUncertaintyImageEnabled = b; }
  void EnableDoseToWaterImage(bool b) { mIsDoseToWaterImageEnabled = b; }
  void EnableDoseToWaterSquaredImage(bool b) { mIsDoseToWaterSquaredImageEnabled = b; }
  void EnableDoseToWaterUncertaintyImage(bool b) { mIsDoseToWaterUncertaintyImageEnabled = b; }
  void EnableNumberOfHitsImage(bool b) { mIsNumberOfHitsImageEnabled = b; }
  void EnableDoseNormalisationToMax(bool b);
  void EnableDoseNormalisationToIntegral(bool b);
  void EnableDoseToWaterNormalisation(bool b) { mIsDoseToWaterNormalisationEnabled = b; mDoseToWaterImage.SetScaleFactor(1.0); }
  void SetDoseAlgorithmType(G4String b) { mDoseAlgorithmType = b; }
  void ImportMassImage(G4String b) { mImportMassImage = b; }
  void ExportMassImage(G4String b) { mExportMassImage = b; }

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
  bool mIsLastHitEventImageEnabled;
  bool mIsEdepImageEnabled;
  
  bool mIsEdepHadElasticImageEnabled;
  bool mIsEdepInelasticImageEnabled;
  bool mIsEdepRestImageEnabled;
  
  bool mIsEdepSquaredImageEnabled;
  bool mIsEdepUncertaintyImageEnabled;
  bool mIsDoseImageEnabled;
  bool mIsDoseSquaredImageEnabled;
  bool mIsDoseUncertaintyImageEnabled;
  bool mIsDoseToWaterImageEnabled;
  bool mIsDoseToWaterSquaredImageEnabled;
  bool mIsDoseToWaterUncertaintyImageEnabled;
  bool mIsNumberOfHitsImageEnabled;
  bool mIsDoseNormalisationEnabled;
  bool mIsDoseToWaterNormalisationEnabled;

  GateImageWithStatistic mEdepImage;
  GateImageWithStatistic mEdepHadElasticImage;
  GateImageWithStatistic mEdepInelasticImage;
  GateImageWithStatistic mEdepRestImage;
  
  GateImageWithStatistic mDoseImage;
  GateImageWithStatistic mDoseToWaterImage;
  GateImageInt mNumberOfHitsImage;
  GateImageInt mLastHitEventImage;
  GateImageDouble mMassImage;

  G4String mEdepFilename;
  G4String mEdepHadElasticFilename;
  G4String mEdepInelasticFilename;
  G4String mEdepRestFilename;
  
  G4String mDoseFilename;
  G4String mDoseToWaterFilename;
  G4String mNbOfHitsFilename;
  G4String mDoseAlgorithmType;
  G4String mImportMassImage;
  G4String mExportMassImage;

  G4EmCalculator* emcalc;
};

MAKE_AUTO_CREATOR_ACTOR(CylindricalEdepActor,GateCylindricalEdepActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
