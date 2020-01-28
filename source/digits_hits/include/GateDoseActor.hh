/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*!
  \class  GateDoseActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr

  - DoseToWater option added by Loïc Grevillot
  - Dose calculation in inhomogeneous volume added by Thomas Deschler (thomas.deschler@iphc.cnrs.fr)
  - Dose in Regions (Maxime Chauvin, David Sarrut)
*/


#ifndef GATEDOSEACTOR_HH
#define GATEDOSEACTOR_HH

#include <G4NistManager.hh>
#include <G4UnitsTable.hh>
#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateDoseActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateVoxelizedMass.hh"
#include "GateRegionDoseStat.hh"

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
  //Edep
  void EnableEdepImage(bool b) { mIsEdepImageEnabled = b; }
  void EnableEdepSquaredImage(bool b) { mIsEdepSquaredImageEnabled = b; }
  void EnableEdepUncertaintyImage(bool b) { mIsEdepUncertaintyImageEnabled = b; }
  //Dose
  void EnableDoseImage(bool b) { mIsDoseImageEnabled = b; }
  void EnableDoseSquaredImage(bool b) { mIsDoseSquaredImageEnabled = b; }
  void EnableDoseUncertaintyImage(bool b) { mIsDoseUncertaintyImageEnabled = b; }
  void EnableDoseNormalisationToMax(bool b);
  void EnableDoseNormalisationToIntegral(bool b);
  void SetEfficiencyFile(G4String b);
  void SetEfficiencyFileByZ(G4String b);
  //DoseToWater
  void EnableDoseToWaterImage(bool b) { mIsDoseToWaterImageEnabled = b; }
  void EnableDoseToWaterSquaredImage(bool b) { mIsDoseToWaterSquaredImageEnabled = b; }
  void EnableDoseToWaterUncertaintyImage(bool b) { mIsDoseToWaterUncertaintyImageEnabled = b; }
  void EnableDoseToWaterNormalisationToMax(bool b);
  void EnableDoseToWaterNormalisationToIntegral(bool b);
  //DoseToOtherMaterial
  void EnableDoseToOtherMaterialImage(bool b) { mIsDoseToOtherMaterialImageEnabled = b; }
  void EnableDoseToOtherMaterialSquaredImage(bool b) { mIsDoseToOtherMaterialSquaredImageEnabled = b; }
  void EnableDoseToOtherMaterialUncertaintyImage(bool b) { mIsDoseToOtherMaterialUncertaintyImageEnabled = b; }
  void EnableDoseToOtherMaterialNormalisationToMax(bool b);
  void EnableDoseToOtherMaterialNormalisationToIntegral(bool b);
  void SetOtherMaterial(G4String b) { mOtherMaterial = b; }
  //Others
  void EnableNumberOfHitsImage(bool b) { mIsNumberOfHitsImageEnabled = b; }
  void SetDoseAlgorithmType(G4String b) { mDoseAlgorithmType = b; }
  void ImportMassImage(G4String b) { mImportMassImage = b; }
  void ExportMassImage(G4String b) { mExportMassImage = b; }
  void VolumeFilter(G4String b) { mVolumeFilter = b; }
  void MaterialFilter(G4String b) { mMaterialFilter = b; }
  void setTestFlag(bool b) { mTestFlag = b; }
  //Regions
  void SetDoseByRegionsInputFilename(std::string f);
  void SetDoseByRegionsOutputFilename(std::string f);
  void AddRegion(std::string str);

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
  GateDoseActor(G4String name, G4int depth=0);
  GateDoseActorMessenger* pMessenger;
  GateVoxelizedMass mVoxelizedMass;

  int mCurrentEvent;
  StepHitType mUserStepHitType;

  bool mIsLastHitEventImageEnabled;

  //Edep
  bool mIsEdepImageEnabled;
  bool mIsEdepSquaredImageEnabled;
  bool mIsEdepUncertaintyImageEnabled;
  //Dose
  bool mIsDoseImageEnabled;
  bool mIsDoseSquaredImageEnabled;
  bool mIsDoseUncertaintyImageEnabled;
  bool mIsDoseNormalisationEnabled;
  bool mIsDoseEfficiencyEnabled;
  bool mIsDoseEfficiencyByZEnabled;
  //DoseToWater
  bool mIsDoseToWaterImageEnabled;
  bool mIsDoseToWaterSquaredImageEnabled;
  bool mIsDoseToWaterUncertaintyImageEnabled;
  bool mIsDoseToWaterNormalisationEnabled;
  bool mDose2WaterWarningFlag;
  //DoseToOtherMaterial
  bool mIsDoseToOtherMaterialImageEnabled;
  bool mIsDoseToOtherMaterialSquaredImageEnabled;
  bool mIsDoseToOtherMaterialUncertaintyImageEnabled;
  bool mIsDoseToOtherMaterialNormalisationEnabled;
  //Others
  bool mIsNumberOfHitsImageEnabled;
  bool mTestFlag;

  //Edep
  G4String mEdepFilename;
  GateImageWithStatistic mEdepImage;
  //Dose
  G4String mDoseFilename;
  GateImageWithStatistic mDoseImage;
    //Efficiency option
  G4String mDoseEfficiencyFile;
  std::vector<double> mDoseEnergy;
  std::vector<double> mDoseEfficiency;
    //Efficiency option by Z (by ion atomic number)
  std::vector<G4String> mDoseEfficiencyFileByZ;
  std::vector<G4int> mDoseZByZ;
  std::vector<std::vector<double>> mDoseEnergyByZ;
  std::vector<std::vector<double>> mDoseEfficiencyByZ;
  //DoseToWater
  G4String mDoseToWaterFilename;
  GateImageWithStatistic mDoseToWaterImage;
  //DoseToOtherMaterial
  G4String mDoseToOtherMaterialFilename;
  GateImageWithStatistic mDoseToOtherMaterialImage;
  G4String mOtherMaterial;
  //Hits
  G4String mNbOfHitsFilename;
  GateImageInt mNumberOfHitsImage;
  GateImageInt mLastHitEventImage;
  //Others
  GateImageDouble mMassImage;
  //Regions
  bool mDoseByRegionsFlag;
  GateImageFloat mDoseByRegionsLabelImage;
  GateRegionDoseStat::IdToSingleRegionMapType mMapIdToSingleRegion;
  GateRegionDoseStat::LabelToSeveralRegionsMapType mMapLabelToSeveralRegions;
  GateRegionDoseStat::IdToLabelsMapType mMapIdToLabels;
  G4String mDoseByRegionsInputFilename;
  G4String mDoseByRegionsOutputFilename;

  G4String mDoseAlgorithmType;
  G4String mImportMassImage;
  G4String mExportMassImage;
  G4String mVolumeFilter;
  G4String mMaterialFilter;

  G4EmCalculator* emcalc;

};

MAKE_AUTO_CREATOR_ACTOR(DoseActor,GateDoseActor)

#endif /* end #define GATESIMULATIONSTATISTICACTOR_HH */
