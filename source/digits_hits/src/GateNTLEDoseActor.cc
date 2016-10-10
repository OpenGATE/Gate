/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateNTLEDoseActor :
  \brief
*/

#include "GateNTLEDoseActor.hh"
#include "GateMiscFunctions.hh"

#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateNTLEDoseActor::GateNTLEDoseActor(G4String name, G4int depth):
  GateVImageActor(name, depth) {
  mCurrentEvent = -1;
  pMessenger = new GateNTLEDoseActorMessenger(this);
  mKFHandler = new GateKermaFactorHandler();

  mIsDoseImageEnabled            = false;
  mIsDoseSquaredImageEnabled     = false;
  mIsDoseUncertaintyImageEnabled = false;
  mIsDoseCorrectionEnabled       = false;
  mIsLastHitEventImageEnabled    = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNTLEDoseActor::~GateNTLEDoseActor() {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::Construct() {
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  if (!mIsDoseImageEnabled)
    GateError("The NTLEDoseActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableDose true' for example)");

  // Output Filename
  mDoseFilename = G4String(removeExtension(mSaveFilename)) + "-Dose." + G4String(getExtension(mSaveFilename));

  SetOriginTransformAndFlagToImage(mDoseImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);

  if (mIsDoseSquaredImageEnabled || mIsDoseUncertaintyImageEnabled) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
    mLastHitEventImage.SetOrigin(mOrigin);
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

  GateMessage("Actor", 1,
              "NTLE DoseActor    = '" << GetObjectName() << "'\n" <<
              "\tDose image        = " << mIsDoseImageEnabled << Gateendl <<
              "\tDose squared      = " << mIsDoseSquaredImageEnabled << Gateendl <<
              "\tDose uncertainty  = " << mIsDoseUncertaintyImageEnabled << Gateendl <<
              "\tDose correction   = " << mIsDoseCorrectionEnabled << Gateendl <<
              "\tDoseFilename      = " << mDoseFilename << Gateendl);

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::SaveData() {
  GateVActor::SaveData();
  if (mIsDoseImageEnabled) mDoseImage.SaveData(mCurrentEvent + 1, false);
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::ResetData() {
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsDoseImageEnabled) mDoseImage.Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::UserSteppingAction(const GateVVolume*, const G4Step* step) {
  const int index = GetIndexFromStepPosition(GetVolume(), step);
  UserSteppingActionInVoxel(index, step);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::BeginOfRunAction(const G4Run* r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateNTLEDoseActor -- Begin of Run\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::BeginOfEventAction(const G4Event* e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateNTLEDoseActor -- Begin of Event: "<< mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  const G4StepPoint* PreStep (step->GetPreStepPoint() );

  if (step->GetTrack()->GetDefinition()->GetParticleName() == "neutron") {
    mKFHandler->SetMaterial(PreStep->GetMaterial());
    mKFHandler->SetCubicVolume(GetDoselVolume());
    mKFHandler->SetDistance(step->GetStepLength());
    mKFHandler->SetEnergy(PreStep->GetKineticEnergy());

    double dose = mKFHandler->GetDose();
    if (mIsDoseCorrectionEnabled)
      dose = mKFHandler->GetDoseCorrected();

    bool sameEvent = true;

    if (mIsLastHitEventImageEnabled) {
      if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
        sameEvent = false;
        mLastHitEventImage.SetValue(index, mCurrentEvent);
      }
    }

    if (mIsDoseImageEnabled) {
      if (mIsDoseUncertaintyImageEnabled || mIsDoseSquaredImageEnabled) {
        if (sameEvent) mDoseImage.AddTempValue(index, dose);
        else mDoseImage.AddValueAndUpdate(index, dose);
      }
      else
        mDoseImage.AddValue(index, dose);
    }
  }
}
//-----------------------------------------------------------------------------
