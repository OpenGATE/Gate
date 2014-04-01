/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \brief Class GateProductionAndStoppingActor :
  \brief
*/

#ifndef GATEPRODANDSTOPACTOR_CC
#define GATEPRODANDSTOPACTOR_CC

#include "GateProductionAndStoppingActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
GateProductionAndStoppingActor::GateProductionAndStoppingActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateProductionAndStoppingActor() -- begin"<<G4endl);

  mCurrentEvent=-1;

  pMessenger = new GateImageActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateProductionAndStoppingActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateProductionAndStoppingActor::~GateProductionAndStoppingActor()  {
  GateDebugMessageInc("Actor",4,"~GateProductionAndStoppingActor() -- begin"<<G4endl);
  GateDebugMessageDec("Actor",4,"~GateProductionAndStoppingActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateProductionAndStoppingActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateProductionAndStoppingActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  // Output Filename
  mProdFilename = G4String(removeExtension(mSaveFilename))+"-Prod."+G4String(getExtension(mSaveFilename));
  mStopFilename = G4String(removeExtension(mSaveFilename))+"-Stop."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mProdImage);
  SetOriginTransformAndFlagToImage(mStopImage);

  mProdImage.EnableSquaredImage(false);
  mProdImage.EnableUncertaintyImage(false);
  mProdImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mProdImage.Allocate();
  mProdImage.SetFilename(mProdFilename);

  mStopImage.EnableSquaredImage(false);
  mStopImage.EnableUncertaintyImage(false);
  mStopImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mStopImage.Allocate();
  mStopImage.SetFilename(mStopFilename);

  ResetData();
  GateMessageDec("Actor", 4, "GateProductionAndStoppingActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateProductionAndStoppingActor::SaveData() {
  mProdImage.SaveData(mCurrentEvent+1);
  mStopImage.SaveData(mCurrentEvent+1);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateProductionAndStoppingActor::ResetData() {
   mProdImage.Reset();
   mStopImage.Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateProductionAndStoppingActor::BeginOfRunAction(const G4Run * ) {
  GateDebugMessage("Actor", 3, "GateProductionAndStoppingActor -- Begin of Run" << G4endl);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateProductionAndStoppingActor::BeginOfEventAction(const G4Event * ) {
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateProductionAndStoppingActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateProductionAndStoppingActor::UserSteppingActionInVoxel(const int index, const G4Step* aStep)
{
  if(index>-1) {
    if ( aStep->GetTrack()->GetCurrentStepNumber() == 1 )  {
      mProdImage.AddValue(index, 1); }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateProductionAndStoppingActor::UserPostTrackActionInVoxel(const int index, const G4Track* /*t*/)
{
  if(index>-1) mStopImage.AddValue(index, 1);
}
//-----------------------------------------------------------------------------


#endif /* end #define GATEPRODANDSTOPACTOR_CC */
