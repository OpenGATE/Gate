/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateFluenceActor : 
  \brief 
*/

#include "GateFluenceActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
GateFluenceActor::GateFluenceActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateFluenceActor() -- begin"<<G4endl);

  pMessenger = new GateFluenceActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateFluenceActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateFluenceActor::~GateFluenceActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateFluenceActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct(); // mImage is not allocated here
  mImage.Allocate();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Allocate scatter image
  if (mIsScatterImageEnabled) {
    mImageScatter.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageScatter.Allocate();
    mImageScatter.SetOrigin(mOrigin);
  }

  // Print information
  GateMessage("Actor", 1, 
              "\tFluence FluenceActor    = '" << GetObjectName() << "'" << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateFluenceActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateFluenceActor::SaveData()
{
  GateVActor::SaveData();

  for(size_t i=0; i<mCounts.size(); i++)
    mImage.SetValue(i, mCounts[i]);
  mImage.Write(mSaveFilename);

  if(mIsScatterImageEnabled) {
    for(size_t i=0; i<mScatterCounts.size(); i++)
      mImageScatter.SetValue(i, mScatterCounts[i]);
    mImageScatter.Write(G4String(removeExtension(mSaveFilename))+"-scatter."+G4String(getExtension(mSaveFilename)));  
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFluenceActor::ResetData() {
  mImage.Fill(0);

  mCounts.resize(mImage.GetNumberOfValues());
  std::fill(mCounts.begin(), mCounts.end(), 0);

  if(mIsScatterImageEnabled) {
    mScatterCounts.resize(mImage.GetNumberOfValues());
    std::fill(mScatterCounts.begin(), mScatterCounts.end(), 0);
    mImageScatter.Fill(0);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFluenceActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel - begin" << G4endl);

  // Is this necessary?
  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  //FIXME
  DD(step->GetPreStepPoint()->GetPosition());
  DD(index);
  DD(mImage.GetCoordinatesFromIndex(index));
  DD(mImage.GetVoxelCenterFromIndex(index));
  

  /* http://geant4.org/geant4/support/faq.shtml
    To check that the particle has just entered in the current volume
    (i.e. it is at the first step in the volume; the preStepPoint is at the boundary):
  */
  if (step->GetPreStepPoint()->GetStepStatus() == fGeomBoundary) {
    mCounts[index]++;
    
    if(mIsScatterImageEnabled &&
       !step->GetTrack()->GetParentID() &&
       !step->GetTrack()->GetDynamicParticle()->GetPrimaryParticle()->GetMomentum().isNear(
            step->GetTrack()->GetDynamicParticle()->GetMomentum())) {
      mScatterCounts[index]++;
    }
  }

  GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------


