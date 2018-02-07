/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \brief Class GateTLEDoseActor :
  \brief
*/

#include "GateTLEDoseActor.hh"
#include "GateMiscFunctions.hh"
#include "GateMaterialMuHandler.hh"

#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateTLEDoseActor::GateTLEDoseActor(G4String name, G4int depth):
  GateVImageActor(name, depth) {
  mCurrentEvent = -1;
  pMessenger = new GateTLEDoseActorMessenger(this);
  mMaterialHandler = GateMaterialMuHandler::GetInstance();
  mIsEdepImageEnabled = false;
  mIsEdepSquaredImageEnabled = false;
  mIsEdepUncertaintyImageEnabled = false;
  mIsDoseSquaredImageEnabled = false;
  mIsDoseUncertaintyImageEnabled = false;
  mIsLastHitEventImageEnabled = false;
  mIsDoseNormalisationEnabled = false;
  mDoseAlgorithmType = "";
  mImportMassImage = "";
  mVolumeFilter = "";
  mMaterialFilter = "";
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateTLEDoseActor::~GateTLEDoseActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActor::EnableDoseNormalisationToMax(bool b) {
  mIsDoseNormalisationEnabled = b;
  mDoseImage.SetNormalizeToMax(b);
  mDoseImage.SetScaleFactor(1.0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActor::EnableDoseNormalisationToIntegral(bool b) {
  mIsDoseNormalisationEnabled = b;
  mDoseImage.SetNormalizeToIntegral(b);
  mDoseImage.SetScaleFactor(1.0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateTLEDoseActor::Construct() {
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  if (!mIsEdepImageEnabled &&
      !mIsDoseImageEnabled) {
    GateError("The TLEDoseActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableEdep true' for example)");
  }

  // Output Filename
  mDoseFilename = G4String(removeExtension(mSaveFilename)) + "-Dose." + G4String(getExtension(mSaveFilename));
  mEdepFilename = G4String(removeExtension(mSaveFilename)) + "-Edep." + G4String(getExtension(mSaveFilename));

  SetOriginTransformAndFlagToImage(mDoseImage);
  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);

  if (mIsEdepSquaredImageEnabled || mIsEdepUncertaintyImageEnabled ||
      mIsDoseSquaredImageEnabled || mIsDoseUncertaintyImageEnabled) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
    mLastHitEventImage.SetOrigin(mOrigin);
  }

  if (mIsEdepImageEnabled) {
    mEdepImage.EnableSquaredImage(mIsEdepSquaredImageEnabled);
    mEdepImage.EnableUncertaintyImage(mIsEdepUncertaintyImageEnabled);
    mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mEdepImage.Allocate();
    mEdepImage.SetFilename(mEdepFilename);
    mEdepImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
    mEdepImage.SetOrigin(mOrigin);
  }

  if (mIsDoseImageEnabled) {
    mDoseImage.EnableSquaredImage(mIsDoseSquaredImageEnabled);
    mDoseImage.EnableUncertaintyImage(mIsDoseUncertaintyImageEnabled);
    mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mDoseImage.Allocate();
    mDoseImage.SetFilename(mDoseFilename);
    mDoseImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
    mDoseImage.SetOrigin(mOrigin);
  }

  if (mIsDoseImageEnabled &&
      (mDoseAlgorithmType == "MassWeighting" || mVolumeFilter != "" || mMaterialFilter != "")) {
    mVoxelizedMass.SetMaterialFilter(mMaterialFilter);
    mVoxelizedMass.SetVolumeFilter(mVolumeFilter);
    mVoxelizedMass.SetExternalMassImage(mImportMassImage);
    mVoxelizedMass.Initialize(mVolumeName, &mDoseImage.GetValueImage());
  }

  ConversionFactor = e_SI * 1.0e11;
  VoxelVolume = GetDoselVolume();
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateTLEDoseActor::SaveData() {
  GateVActor::SaveData();
  if (mIsDoseImageEnabled) {
    if (mIsDoseNormalisationEnabled)
      mDoseImage.SaveData(mCurrentEvent+1, true);
    else
      mDoseImage.SaveData(mCurrentEvent+1, false);
  }
  if (mIsEdepImageEnabled) mEdepImage.SaveData(mCurrentEvent + 1, false);
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActor::ResetData() {
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsEdepImageEnabled) mEdepImage.Reset();
  if (mIsDoseImageEnabled) mDoseImage.Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActor::UserSteppingAction(const GateVVolume *, const G4Step *step)
{
  int index = GetIndexFromStepPosition(GetVolume(), step);
  UserSteppingActionInVoxel(index, step);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActor::BeginOfRunAction(const G4Run *r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateDoseActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback at each event
void GateTLEDoseActor::BeginOfEventAction(const G4Event *e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActor::UserSteppingActionInVoxel(const int index, const G4Step *step) {
  G4StepPoint *PreStep(step->GetPreStepPoint());
  G4StepPoint *PostStep(step->GetPostStepPoint());
  G4ThreeVector prePosition = PreStep->GetPosition();
  G4ThreeVector postPosition = PostStep->GetPosition();

  if (step->GetTrack()->GetDefinition()->GetParticleName() == "gamma") {
    // Filters conditions
    if ((mVolumeFilter != "" && mVolumeFilter+"_phys" != step->GetPreStepPoint()->GetPhysicalVolume()->GetName()) ||
        (mMaterialFilter != "" && mMaterialFilter != step->GetPreStepPoint()->GetMaterial()->GetName()))
      return;

    double distance = step->GetStepLength();
    double energy = PreStep->GetKineticEnergy();
    double muenOverRho = mMaterialHandler->GetMuEnOverRho(PreStep->GetMaterialCutsCouple(), energy);
    double dose = ConversionFactor * energy * muenOverRho * distance / VoxelVolume;

    //---------------------------------------------------------------------------------
    // Mass weighting OR filter
    if (mDoseAlgorithmType == "MassWeighting" || mMaterialFilter != "" || mVolumeFilter != "") {
      double muen = mMaterialHandler->GetMuEn(PreStep->GetMaterialCutsCouple(), energy);
      dose = energy * muen * distance / mVoxelizedMass.GetDoselMass(index) / gray * 0.1;
    }
    //---------------------------------------------------------------------------------

    double edep = 0.1 * energy * muenOverRho * distance * PreStep->GetMaterial()->GetDensity() / (g / cm3);
    bool sameEvent = true;

    if (mIsLastHitEventImageEnabled) {
      if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
        sameEvent = false;
        mLastHitEventImage.SetValue(index, mCurrentEvent);
      }
    }

    if (energy <= .001) {
      edep = energy;
      step->GetTrack()->SetTrackStatus(fStopAndKill);
    }

    if (mIsDoseImageEnabled) {
      if (mIsDoseUncertaintyImageEnabled || mIsDoseSquaredImageEnabled) {
        if (sameEvent) mDoseImage.AddTempValue(index, dose);
        else mDoseImage.AddValueAndUpdate(index, dose);
      }
      else
        mDoseImage.AddValue(index, dose);
    }
    if (mIsEdepImageEnabled) {
      if (mIsEdepUncertaintyImageEnabled || mIsEdepSquaredImageEnabled) {
        if (sameEvent) mEdepImage.AddTempValue(index, edep);
        else mEdepImage.AddValueAndUpdate(index, edep);
      }
      else
        mEdepImage.AddValue(index, edep);
    }

  }
}
//-----------------------------------------------------------------------------
