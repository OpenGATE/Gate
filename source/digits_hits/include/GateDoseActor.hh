/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateDoseActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr

	  DoseToWater option added by Loïc Grevillot
  \date	March 2011
 */

#ifndef GATEDOSEACTOR_HH
#define GATEDOSEACTOR_HH

#include <G4NistManager.hh>

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateDoseActorMessenger.hh"
#include "GateImageWithStatistic.hh"

class G4EmCalculator;

class GateDoseActor : public GateVImageActor
{
 public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateDoseActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateDoseActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  void EnableEdepImage(bool b) { mIsEdepImageEnabled = b; }
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

  void EnableRBE1AlphaImage(bool b) { mIsRBE1AlphaImageEnabled = b; }
  void EnableRBE1BetaImage(bool b) { mIsRBE1BetaImageEnabled = b; }
  void EnableRBE1FactorImage(bool b) { mIsRBE1FactorImageEnabled = b; }
  void EnableRBE1BioDoseImage(bool b) { mIsRBE1BioDoseImageEnabled = b; }
  void EnableRBE1Test1(bool b) { mIsRBE1Test1Enabled = b; }

  void SetRBE1AlphaDataFilename(G4String f) { mRBE1AlphaDataFilename = f; }
  void SetRBE1BetaDataFilename(G4String f) { mRBE1BetaDataFilename = f; }

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void BeginOfEventAction(const G4Event * event);

  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track);
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  ///Scorer related
  //virtual G4bool ProcessHits(G4Step *, G4TouchableHistory*);
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}

  void ReadRBE1AlphaBetaFromFile(G4String filenameAlpha, G4String filenameBeta);
  void GetRBE1AlphaBetaFromLet(G4double let, G4double & alpha, G4double & beta);
  void ComputeRBE1ImageAndSave();

protected:
  GateDoseActor(G4String name, G4int depth=0);
  GateDoseActorMessenger * pMessenger;

  int mCurrentEvent;
  StepHitType mUserStepHitType;

  bool mIsLastHitEventImageEnabled;
  bool mIsEdepImageEnabled;
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
  bool mIsRBE1AlphaImageEnabled;
  bool mIsRBE1BetaImageEnabled;
  bool mIsRBE1FactorImageEnabled;
  bool mIsRBE1BioDoseImageEnabled;
  bool mIsRBE1Enabled;
  bool mIsRBE1Test1Enabled;

  GateImageWithStatistic mEdepImage;
  GateImageWithStatistic mDoseImage;
  GateImageWithStatistic mDoseToWaterImage;
  GateImage mNumberOfHitsImage;
  GateImage mLastHitEventImage;
  GateImageWithStatistic mRBE1AlphaImage;
  GateImageWithStatistic mRBE1BetaImage;
  GateImageWithStatistic mRBE1FactorImage;
  GateImageWithStatistic mRBE1BioDoseImage;

  G4String mEdepFilename;
  G4String mDoseFilename;
  G4String mDoseToWaterFilename;
  G4String mNbOfHitsFilename;
  G4String mRBE1AlphaFilename;
  G4String mRBE1BetaFilename;
  G4String mRBE1AlphaDataFilename;
  G4String mRBE1BetaDataFilename;
  G4String mRBE1FactorFilename;
  G4String mRBE1BioDoseFilename;

  std::vector<G4double> mAlphaLet;
  std::vector<G4double> mAlphaValues;
  std::vector<G4double> mBetaLet;
  std::vector<G4double> mBetaValues;

  G4EmCalculator * emcalc;
};

MAKE_AUTO_CREATOR_ACTOR(DoseActor,GateDoseActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
