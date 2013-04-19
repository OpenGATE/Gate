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
// Gate
#include "GateFluenceActor.hh"
#include "GateScatterOrderTrackInformationActor.hh"

//-----------------------------------------------------------------------------
GateFluenceActor::GateFluenceActor(G4String name, G4int depth):
  GateVImageActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateFluenceActor() -- begin"<<G4endl);
  pMessenger = new GateFluenceActorMessenger(this);
  mCurrentEventFrame = 0;
  mUseFrameFlag = false;
  mIsSourceMotionByStackEnabled = false;
  SetStepHitType("pre");
  GateDebugMessageDec("Actor",4,"GateFluenceActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateFluenceActor::~GateFluenceActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateFluenceActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct(); // mImage is not allocated here
  mImage.Allocate();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // the image index will be computed according to the preStep
  if (mStepHitType != PreStepHitType) {
    GateWarning("The stepHitType must be 'pre', we force it.");
    SetStepHitType("pre");
  }
  SetStepHitType("pre");

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
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  char filename[1024];
  // Printing all particles
  GateVImageActor::SaveData();
  if(mSaveFilename != "")
  {
    sprintf(filename, mSaveFilename, rID);
    mImage.Write(filename);
    // Printing just scatter
    if(mIsScatterImageEnabled)
    {
      G4String fn = removeExtension(filename)+"-scatter."+G4String(getExtension(filename));
      mImageScatter.Write(fn);
    }
  }
  // Printing scatter of each order
  if(mScatterOrderFilename != "")
  {
    for(unsigned int k = 0; k<mFluencePerOrderImages.size(); k++)
    {
      sprintf(filename, mScatterOrderFilename, rID, k+1);
      mFluencePerOrderImages[k]->Write((G4String)filename);
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::ResetData() 
{
  mImage.Fill(0);
  if(mIsScatterImageEnabled) {
    mImageScatter.Fill(0);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::BeginOfEventAction(const G4Event * e)
{ 
  GateVActor::BeginOfEventAction(e);

  if (mIsSourceMotionByStackEnabled && e->GetUserInformation()) {
    // Retrieve current stack in the event info
    GateEventSourceMoveRandomInformation * info = 
      static_cast<GateEventSourceMoveRandomInformation *>(e->GetUserInformation());
    mCurrentEventFrame = info->index;
    
    // If this is the first time we know that we need to change image
    // dimension
    if (!mUseFrameFlag) {
      mDimension = -1;
      if (mResolution.x() == 1) mDimension = 0;
      if (mResolution.y() == 1) mDimension = 1;
      if (mResolution.z() == 1) mDimension = 2;
      if (mDimension == -1) {
        GateError("You must have a dimension ==1 to use this actor with dynamic source");
      }
      G4ThreeVector r = mResolution;
      r[mDimension] = info->nbOfFrame;
      mImage.SetResolutionAndHalfSize(r, mHalfSize, mPosition);
      mImage.Allocate();
      if(mIsScatterImageEnabled) {
        mImageScatter.SetResolutionAndHalfSize(r, mHalfSize, mPosition);
        mImageScatter.Allocate();
      }
      mUseFrameFlag = true;
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActor::UserSteppingActionInVoxel(const int index, const G4Step* step)
{
  GateDebugMessageInc("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel - begin" << G4endl);

  // Is this necessary?
  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  // If needed, take into account the event index to store value into
  // the corresponding frame
  int newIndex = index;
  if (mUseFrameFlag) {
    /*
      DD("-----------------------*******************************");
      DD(mStepHitType);
      DD(mCurrentEventFrame);
      DD(step->GetPreStepPoint()->GetPosition());
      DD(step->GetPostStepPoint()->GetPosition());
      DD(index);
      DD(mImage.GetCoordinatesFromIndex(index));
      DD(step->GetPreStepPoint()->GetStepStatus());
      DD(step->GetPostStepPoint()->GetStepStatus());
      DD(fGeomBoundary);
    */
    G4ThreeVector c = mImage.GetCoordinatesFromIndex(index);
    c[mDimension] = mCurrentEventFrame;
    newIndex = mImage.GetIndexFromCoordinates(c);
    G4ThreeVector d = mImage.GetCoordinatesFromIndex(newIndex);
  }
  
  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());

  /* http://geant4.org/geant4/support/faq.shtml
     To check that the particle has just entered in the current volume
     (i.e. it is at the first step in the volume; the preStepPoint is at the boundary):
  */
  if (step->GetPreStepPoint()->GetStepStatus() == fGeomBoundary) {
    mImage.AddValue(newIndex, 1);
    // Scatter order
    if(info)
    {
      unsigned int order = info->GetScatterOrder();
      if(order)
      {
        while(order>mFluencePerOrderImages.size() && order>0)
        {
          GateImage * voidImage = new GateImage;
          voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
          voidImage->Allocate();
          voidImage->SetOrigin(mOrigin);
          voidImage->Fill(0);
          mFluencePerOrderImages.push_back( voidImage );
        }
          mFluencePerOrderImages[order-1]->AddValue(newIndex, 1);
      }
    }
    
    if(mIsScatterImageEnabled &&
       !step->GetTrack()->GetParentID() &&
       !step->GetTrack()->GetDynamicParticle()->GetPrimaryParticle()->GetMomentum().isNear(
                                                                                           step->GetTrack()->GetDynamicParticle()->GetMomentum())) {
        mImageScatter.AddValue(newIndex, 1);
    }
  }

  GateDebugMessageDec("Actor", 4, "GateFluenceActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------
