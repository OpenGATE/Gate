/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateStoppingPowerActor :
  \brief
*/

#include "GateDeltaEnergyActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
GateStoppingPowerActor::GateStoppingPowerActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateStoppingPowerActor() -- begin"<<G4endl);

  mCurrentEvent=-1;

  mIsStopPowerImageEnabled = true;
  mIsRelStopPowerImageEnabled = true;

  mIsLastHitEventImageEnabled = false;
  mIsNumberOfHitsImageEnabled = true;

  mIsStopPowerSquaredImageEnabled = false;
  mIsStopPowerUncertaintyImageEnabled = false;
  mIsRelStopPowerSquaredImageEnabled = false;
  mIsRelStopPowerUncertaintyImageEnabled = false;

  pMessenger = new GateImageActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateStoppingPowerActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateStoppingPowerActor::~GateStoppingPowerActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateStoppingPowerActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateStoppingPowerActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Check if at least one image is enabled
  if (!mIsStopPowerImageEnabled &&
      !mIsRelStopPowerImageEnabled &&
      !mIsNumberOfHitsImageEnabled ){
    GateError("The StoppingPowerActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableStopPower true' for example)");
  }

  SetOriginTransformAndFlagToImage(mStopPowerImage);
  SetOriginTransformAndFlagToImage(mRelStopPowerImage);
  SetOriginTransformAndFlagToImage(mNumberOfHitsImage);
  SetOriginTransformAndFlagToImage(mLastHitEventImage);

  // Output Filename
  mStopPowerFilename = G4String(removeExtension(mSaveFilename))+"-StopPower."+G4String(getExtension(mSaveFilename));
  mRelStopPowerFilename = G4String(removeExtension(mSaveFilename))+"-RelStopPower."+G4String(getExtension(mSaveFilename));
  mNbOfHitsFilename = G4String(removeExtension(mSaveFilename))+"-NbOfHits."+G4String(getExtension(mSaveFilename));

  // Resize and allocate images
  if (mIsStopPowerSquaredImageEnabled || mIsStopPowerUncertaintyImageEnabled ||
      mIsRelStopPowerSquaredImageEnabled || mIsRelStopPowerUncertaintyImageEnabled) {
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
  }
  if (mIsStopPowerImageEnabled) {
    //  mStopPowerImage.SetLastHitEventImage(&mLastHitEventImage);
    mStopPowerImage.EnableSquaredImage(mIsStopPowerSquaredImageEnabled);
    mStopPowerImage.EnableUncertaintyImage(mIsStopPowerUncertaintyImageEnabled);
    mStopPowerImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mStopPowerImage.Allocate();
    mStopPowerImage.SetFilename(mStopPowerFilename);
    mStopPowerImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
  }
  if (mIsRelStopPowerImageEnabled) {
    // mRelStopPowerImage.SetLastHitEventImage(&mLastHitEventImage);
    mRelStopPowerImage.EnableSquaredImage(mIsRelStopPowerSquaredImageEnabled);
    mRelStopPowerImage.EnableUncertaintyImage(mIsRelStopPowerUncertaintyImageEnabled);
    mRelStopPowerImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    // DD(mRelStopPowerImage.GetVoxelVolume());
    //mRelStopPowerImage.SetScaleFactor(1e12/mRelStopPowerImage.GetVoxelVolume());
    mRelStopPowerImage.Allocate();
    mRelStopPowerImage.SetFilename(mRelStopPowerFilename);
    mRelStopPowerImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
  }
  if (mIsNumberOfHitsImageEnabled) {
    mNumberOfHitsImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mNumberOfHitsImage.Allocate();
  }

  // Print information
  GateMessage("Actor", 1,
              "\tRelStopPower StoppingPowerActor    = '" << GetObjectName() << "'" << G4endl <<
              "\tRelStopPower image        = " << mIsRelStopPowerImageEnabled << G4endl <<
              "\tRelStopPower squared      = " << mIsRelStopPowerSquaredImageEnabled << G4endl <<
              "\tRelStopPower uncertainty  = " << mIsRelStopPowerUncertaintyImageEnabled << G4endl <<
              "\tStopPower image        = " << mIsStopPowerImageEnabled << G4endl <<
              "\tStopPower squared      = " << mIsStopPowerSquaredImageEnabled << G4endl <<
              "\tStopPower uncertainty  = " << mIsStopPowerUncertaintyImageEnabled << G4endl <<
              "\tNumber of hit     = " << mIsNumberOfHitsImageEnabled << G4endl <<
              "\t     (last hit)   = " << mIsLastHitEventImageEnabled << G4endl <<
              "\tNb Hits filename  = " << mNbOfHitsFilename << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateStoppingPowerActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateStoppingPowerActor::SaveData() {
  GateVActor::SaveData();

  if (mIsStopPowerImageEnabled) mStopPowerImage.SaveData(mCurrentEvent+1);

  if (mIsRelStopPowerImageEnabled) mRelStopPowerImage.SaveData(mCurrentEvent+1, false);

  if (mIsLastHitEventImageEnabled) {
    mLastHitEventImage.Fill(-1); // reset
  }

  if (mIsNumberOfHitsImageEnabled) {
    G4String a = GetSaveCurrentFilename(mNbOfHitsFilename);
    mNumberOfHitsImage.Write(a);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateStoppingPowerActor::ResetData() {
  if (mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
  if (mIsStopPowerImageEnabled) mStopPowerImage.Reset();
  if (mIsRelStopPowerImageEnabled) mRelStopPowerImage.Reset();
  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.Fill(0);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//G4bool GateStoppingPowerActor::ProcessHits(G4Step * step , G4TouchableHistory* th)
void GateStoppingPowerActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  int index = GetIndexFromStepPosition(GetVolume(), step);
  UserSteppingActionInVoxel(index, step);
  //return true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateStoppingPowerActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateStoppingPowerActor -- Begin of Run" << G4endl);
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateStoppingPowerActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateStoppingPowerActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateStoppingPowerActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateStoppingPowerActor -- UserSteppingActionInVoxel - begin" << G4endl);

  // double weight = step->GetTrack()->GetWeight(); // unused yet. Keep it for debug
  double stoppingPower = step->GetPreStepPoint()->GetKineticEnergy() - step->GetPostStepPoint()->GetKineticEnergy() ;
  double kineE = step->GetPreStepPoint()->GetKineticEnergy();
  double relativeStopPower = stoppingPower/kineE;

  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateStoppingPowerActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  bool sameEvent=true;
  if (mIsLastHitEventImageEnabled) {
    GateDebugMessage("Actor", 2,  "GateStoppingPowerActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
    if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
      sameEvent = false;
      mLastHitEventImage.SetValue(index, mCurrentEvent);
    }
  }

  if (mIsRelStopPowerImageEnabled) {

    if (mIsRelStopPowerUncertaintyImageEnabled || mIsRelStopPowerSquaredImageEnabled) {
      if (sameEvent) mRelStopPowerImage.AddTempValue(index, relativeStopPower);
      else mRelStopPowerImage.AddValueAndUpdate(index, relativeStopPower);
    }
    else mRelStopPowerImage.AddValue(index, relativeStopPower);
  }

  if (mIsStopPowerImageEnabled) {
    if (mIsStopPowerUncertaintyImageEnabled || mIsStopPowerSquaredImageEnabled) {
      if (sameEvent) mStopPowerImage.AddTempValue(index, stoppingPower);
      else mStopPowerImage.AddValueAndUpdate(index, stoppingPower);
    }
    else mStopPowerImage.AddValue(index, stoppingPower);
  }

  if (mIsNumberOfHitsImageEnabled) mNumberOfHitsImage.AddValue(index, 1);




  GateDebugMessageDec("Actor", 4, "GateStoppingPowerActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------
