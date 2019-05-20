/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*
  \brief Class GateParticleInVolumeActor :
  \brief
*/


#include "GateParticleInVolumeActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
GateParticleInVolumeActor::GateParticleInVolumeActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateParticleInVolumeActor() -- begin\n");
  outsideTrack = false;
  mCurrentEvent = -1;
  mIsParticleInVolumeImageEnabled = true;
  pMessenger = new GateImageActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateParticleInVolumeActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateParticleInVolumeActor::~GateParticleInVolumeActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateParticleInVolumeActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateParticleInVolumeActor -- Construct - begin\n");
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);

  SetOriginTransformAndFlagToImage(mLastHitEventImage);
  SetOriginTransformAndFlagToImage(mParticleInVolumeImage);

  // Resize and allocate images
  mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mLastHitEventImage.Allocate();
  mIsLastHitEventImageEnabled = true;

  mParticleInVolumeImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mParticleInVolumeImage.Allocate();

  ResetData();
  GateMessageDec("Actor", 4, "GateParticleInVolumeActor -- Construct - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateParticleInVolumeActor::SaveData() {
  GateVActor::SaveData();
  mParticleInVolumeImage.Write(mSaveFilename);
  mLastHitEventImage.Fill(-1); // reset
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateParticleInVolumeActor::ResetData() {
  mParticleInVolumeImage.Fill(0);
  mLastHitEventImage.Fill(-1);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateParticleInVolumeActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateParticleInVolumeActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateParticleInVolumeActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);

  mLastHitEventImage.Fill(-1);

  ++mCurrentEvent;
  GateDebugMessage("Actor", 3, "GateParticleInVolumeActor -- Begin of Event: "<<mCurrentEvent << Gateendl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateParticleInVolumeActor::UserPreTrackActionInVoxel(const int index, const G4Track* /*t*/){
  if(index<0) outsideTrack=true;
  else outsideTrack=false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateParticleInVolumeActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateParticleInVolumeActor -- UserSteppingActionInVoxel - begin\n");

  // double weight = step->GetTrack()->GetWeight(); // unused yet. Keep it for debug

  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing\n");
    GateDebugMessageDec("Actor", 4, "GateParticleInVolumeActor -- UserSteppingActionInVoxel -- end\n");
    return;
  }

  bool sameTrack=true;
  if(outsideTrack){
    int id = step->GetTrack()->GetTrackID();
    if (id != mLastHitEventImage.GetValue(index)) {
      sameTrack = false;
      mLastHitEventImage.SetValue(index, id);
    }
  }

  if(!sameTrack){
    mParticleInVolumeImage.AddValue(index, 1);
  }

  GateDebugMessageDec("Actor", 4, "GateParticleInVolumeActor -- UserSteppingActionInVoxel -- end\n");
}
//-----------------------------------------------------------------------------
