/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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
  GateDebugMessageInc("Actor",4,"GateProductionAndStoppingActor() -- begin\n");

  mCurrentEvent=-1;

  pMessenger = new GateProductionAndStoppingActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateProductionAndStoppingActor() -- end\n");

  bEnableCoordFrame=false;
  bCoordFrame = " ";
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateProductionAndStoppingActor::~GateProductionAndStoppingActor()  {
  GateDebugMessageInc("Actor",4,"~GateProductionAndStoppingActor() -- begin\n");
  GateDebugMessageDec("Actor",4,"~GateProductionAndStoppingActor() -- end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateProductionAndStoppingActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateProductionAndStoppingActor -- Construct - begin\n");
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

/*TODO BRENT
  if (customfr) mProdImage.SetTransformMatrix(RxM);*/

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
  GateMessageDec("Actor", 4, "GateProductionAndStoppingActor -- Construct - end\n");
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
  GateDebugMessage("Actor", 3, "GateProductionAndStoppingActor -- Begin of Run\n");
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateProductionAndStoppingActor::BeginOfEventAction(const G4Event * ) {
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateProductionAndStoppingActor -- Begin of Event: "<<mCurrentEvent << Gateendl);
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
