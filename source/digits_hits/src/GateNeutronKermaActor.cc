/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateNeutronKermaActor :
  \brief
*/

#include "GateNeutronKermaActor.hh"
#include "GateMiscFunctions.hh"
#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>
#include <G4VProcess.hh>

//-----------------------------------------------------------------------------
GateNeutronKermaActor::GateNeutronKermaActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateNeutronKermaActor() -- begin\n");

  mCurrentEvent=-1;
  mIsLastHitEventImageEnabled = false;

  mIsEdepImageEnabled             = false;
  mIsEdepSquaredImageEnabled      = false;
  mIsEdepUncertaintyImageEnabled  = false;

  mIsDoseImageEnabled             = false;
  mIsDoseSquaredImageEnabled      = false;
  mIsDoseUncertaintyImageEnabled  = false;

  mIsNumberOfHitsImageEnabled     = false;
  mIsDoseNormalisationEnabled     = false;

  pMessenger = new GateNeutronKermaActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateNeutronKermaActor() -- end\n");
  emcalc = new G4EmCalculator;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNeutronKermaActor::~GateNeutronKermaActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateNeutronKermaActor -- Construct - begin\n");
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction      (true);
  EnableBeginOfEventAction    (true);
  EnablePreUserTrackingAction (false);
  EnableUserSteppingAction    (true);

  // Check if at least one image is enabled
  if (!mIsEdepImageEnabled &&
      !mIsDoseImageEnabled &&
      !mIsNumberOfHitsImageEnabled)
    GateError("The KermaActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableEdep true' for example)");

  // Output Filename
  mEdepFilename     = G4String(removeExtension(mSaveFilename))+"-Edep."+G4String(getExtension(mSaveFilename));
  mDoseFilename     = G4String(removeExtension(mSaveFilename))+"-Dose."+G4String(getExtension(mSaveFilename));
  mNbOfHitsFilename = G4String(removeExtension(mSaveFilename))+"-NbOfHits."+G4String(getExtension(mSaveFilename));

  SetOriginTransformAndFlagToImage(mEdepImage);
  SetOriginTransformAndFlagToImage(mDoseImage);
  SetOriginTransformAndFlagToImage(mNumberOfHitsImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);

  // Resize and allocate images
  if (mIsEdepSquaredImageEnabled || mIsEdepUncertaintyImageEnabled ||
      mIsDoseSquaredImageEnabled || mIsDoseUncertaintyImageEnabled) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
  }
  if (mIsEdepImageEnabled) {
    if (mIsEdepUncertaintyImageEnabled) mEdepImage.EnableSquaredImage(true);
    mEdepImage.EnableSquaredImage(mIsEdepSquaredImageEnabled);
    mEdepImage.EnableUncertaintyImage(mIsEdepUncertaintyImageEnabled);
    mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);

    mEdepImage.Allocate();
    mEdepImage.SetFilename(mEdepFilename);
  }

  if (mIsDoseImageEnabled) {
    if (mIsDoseUncertaintyImageEnabled) mDoseImage.EnableSquaredImage(true);
    mDoseImage.EnableSquaredImage(mIsDoseSquaredImageEnabled);
    mDoseImage.EnableUncertaintyImage(mIsDoseUncertaintyImageEnabled);
    mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);

    mDoseImage.Allocate();
    mDoseImage.SetFilename(mDoseFilename);
  }

  if (mIsNumberOfHitsImageEnabled) {
    mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mNumberOfHitsImage.Allocate();
  }

  // Print information
  GateMessage("Actor", 1,
              "\tDose KermaActor    = '" << GetObjectName() << "'\n" <<
              "\tDose image        = " << mIsDoseImageEnabled << Gateendl <<
              "\tDose squared      = " << mIsDoseSquaredImageEnabled << Gateendl <<
              "\tDose uncertainty  = " << mIsDoseUncertaintyImageEnabled << Gateendl <<
              "\tEdep image        = " << mIsEdepImageEnabled << Gateendl <<
              "\tEdep squared      = " << mIsEdepSquaredImageEnabled << Gateendl <<
              "\tEdep uncertainty  = " << mIsEdepUncertaintyImageEnabled << Gateendl <<
              "\tNumber of hit     = " << mIsNumberOfHitsImageEnabled << Gateendl <<
              "\t     (last hit)   = " << mIsLastHitEventImageEnabled << Gateendl <<
              "\tedepFilename      = " << mEdepFilename << Gateendl <<
              "\tdoseFilename      = " << mDoseFilename << Gateendl <<
              "\tNb Hits filename  = " << mNbOfHitsFilename << Gateendl);

  ResetData();
  GateMessageDec("Actor", 4, "GateNeutronKermaActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActor::SaveData() {
  GateVActor::SaveData(); // (not needed because done into GateImageWithStatistic)

  if (mIsEdepImageEnabled) mEdepImage.SaveData(mCurrentEvent+1);
  if (mIsDoseImageEnabled) {
    if (mIsDoseNormalisationEnabled)
      mDoseImage.SaveData(mCurrentEvent+1, true);
    else
      mDoseImage.SaveData(mCurrentEvent+1, false);
  }


  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.Write(mNbOfHitsFilename);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActor::ResetData() {
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsEdepImageEnabled) mEdepImage.Reset();
  if (mIsDoseImageEnabled) mDoseImage.Reset();
  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.Fill(0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateNeutronKermaActor -- Begin of Run\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateNeutronKermaActor -- Begin of Event: "<<mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateMessageInc("Actor", 4, "GateNeutronKermaActor -- UserSteppingActionInVoxel - begin\n");

  const double weight = step->GetTrack()->GetWeight();

  GateDebugMessage("Actor", 4, "weight = " << weight << Gateendl);

  double edep = 0.;
  double dose = 0.;

  if (step->GetTrack()->GetDefinition()->GetParticleName() == "neutron")
    edep = step->GetPreStepPoint()->GetKineticEnergy() - step->GetPostStepPoint()->GetKineticEnergy();
  else
    return;

  if (edep <= 0.) {
    GateMessage("Actor", 5, "edep <= 0 : do nothing\n");
    GateMessageDec("Actor", 4, "GateNeutronKermaActor -- UserSteppingActionInVoxel -- end\n");
    return;
  }
  if (index < 0) {
    GateMessage("Actor", 5, "index < 0 : do nothing\n");
    GateMessageDec("Actor", 4, "GateNeutronKermaActor -- UserSteppingActionInVoxel -- end\n");
    return;
  }

  GateMessage("Actor", 2, "GateNeutronKermaActor -- UserSteppingActionInVoxel: edep = " << G4BestUnit(edep, "Energy") << ", PreEKin = " << G4BestUnit(step->GetPreStepPoint()->GetKineticEnergy(), "Energy") << ", PostEKin = " << G4BestUnit(step->GetPostStepPoint()->GetKineticEnergy(), "Energy") << ", ProcName: " << (G4String)step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << Gateendl);

  // compute sameEvent
  // sameEvent is false the first time some energy is deposited for each primary particle
  bool sameEvent = true;
  if (mIsLastHitEventImageEnabled) {
    GateDebugMessage("Actor", 2,  "GateNeutronKermaActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << Gateendl);
    if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
      sameEvent = false;
      mLastHitEventImage.SetValue(index, mCurrentEvent);
    }
  }


  if (mIsDoseImageEnabled) {
    double density = step->GetPreStepPoint()->GetMaterial()->GetDensity();

    // ------------------------------------
    // Convert deposited energy into Gray
    dose = edep/density/mDoseImage.GetVoxelVolume()/gray;
    // ------------------------------------

    GateDebugMessage("Actor", 2,  "GateNeutronKermaActor -- UserSteppingActionInVoxel:"
         << "\tdose = " << G4BestUnit(dose, "Dose")
		     << " rho = "   << G4BestUnit(density, "Volumic Mass")<< Gateendl );
  }

  if (mIsDoseImageEnabled) {

    if (mIsDoseUncertaintyImageEnabled || mIsDoseSquaredImageEnabled) {
      if (sameEvent) mDoseImage.AddTempValue(index, dose);
      else mDoseImage.AddValueAndUpdate(index, dose);
    }
    else mDoseImage.AddValue(index, dose);
  }

  if (mIsEdepImageEnabled) {
    if (mIsEdepUncertaintyImageEnabled || mIsEdepSquaredImageEnabled) {
      if (sameEvent) mEdepImage.AddTempValue(index, edep);
      else mEdepImage.AddValueAndUpdate(index, edep);
    }
    else mEdepImage.AddValue(index, edep);
  }

  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.AddValue(index, weight);

  GateDebugMessageDec("Actor", 4, "GateNeutronKermaActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
