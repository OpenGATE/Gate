/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEVIMAGEACTOR_CC
#define GATEVIMAGEACTOR_CC

#include "G4Step.hh"
//#include "G4UnitsTable.hh"

#include "GateVImageActor.hh"

#include "GateMiscFunctions.hh"
#include "GateObjectStore.hh"
#include "G4TouchableHistory.hh"

//-----------------------------------------------------------------------------
/// Constructor 
GateVImageActor::GateVImageActor(G4String name, G4int depth): 
  GateVActor(name,depth),
  mVoxelSize(-1.0, -1.0, -1.0),
  mResolution(-1.0,-1.0,-1.0),
  mHalfSize(0.0, 0.0, 0.0),
  mPosition(0.0, 0.0, 0.0),
  mStepHitType(MiddleStepHitType),
  mStepHitTypeName("middle"),
  mVoxelSizeIsSet(false),
  mResolutionIsSet(false), 
  mHalfSizeIsSet(false), 
  mPositionIsSet(false)
{
  GateMessageInc("Actor",4, "GateVImageActor() - begin"<<G4endl);
  //pMessenger = new GateImageActorMessenger(this);
  GateMessageDec("Actor",4, "GateVImageActor() - end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor 
GateVImageActor::~GateVImageActor()
{
  GateMessageInc("Actor",4, "~GateVImageActor() - begin"<<G4endl);
  //if (pMessenger) delete pMessenger;
  GateMessageDec("Actor",4, "~GateVImageActor() - end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::SetResolution(G4ThreeVector v) 
{
  mResolutionIsSet = true; 
  mResolution = v; 
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::SetVoxelSize(G4ThreeVector v)
{ 
  mVoxelSizeIsSet = true; 
  mVoxelSize = v;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::SetHalfSize(G4ThreeVector v)
{ 
  mHalfSizeIsSet = true; 
  mHalfSize.setX(v.x());
  mHalfSize.setY(v.y());
  mHalfSize.setZ(v.z());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::SetSize(G4ThreeVector v)
{ 
  mHalfSizeIsSet = true; 
  mHalfSize.setX(v.x()/2.0);
  mHalfSize.setY(v.y()/2.0);
  mHalfSize.setZ(v.z()/2.0);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::SetPosition(G4ThreeVector v)
{ 
  mPositionIsSet = true;
  mPosition = v;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/*void GateVImageActor::SetPosition(GateVVolume * v)
  { 
  G4ThreeVector vPos = v->GetPhysicalVolume()->GetObjectTranslation();

  if(mPositionIsSet == true)
  {
  mPosition.setX( mPosition.x() +vPos.x());
  mPosition.setY( mPosition.y() +vPos.y());
  mPosition.setZ( mPosition.z() +vPos.z());
  }
  else 
  {
  mPosition.setX(vPos.x());
  mPosition.setY(vPos.y());
  mPosition.setZ(vPos.z()); 
  }

  mPositionIsSet = true;
  }*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Constructs the sensor 
void GateVImageActor::Construct() 
{ 
  GateDebugMessageInc("Actor", 4, "GateVImageActor -- Construct: begin" << G4endl);
  GateVActor::Construct();

  if (!mHalfSizeIsSet){
    mHalfSize = ComputeBoundingBox(mVolume->GetLogicalVolume()->GetSolid());  
  }
 
  //if (mPosition.x() == 0 && 
  //	  mPosition.y() == 0 && 
  //  mPosition.z() == 0) { mPositionIsSet = false; }

  if (mResolutionIsSet && mVoxelSizeIsSet) {
    GateError("GateVImageActor -- Construct: Please give the resolution OR the voxelsize (not both) for the sensor");
  }

  if (!mResolutionIsSet && !mVoxelSizeIsSet) {
    mResolution = G4ThreeVector(1.0, 1.0, 1.0);
    mResolutionIsSet = true;
  }

  GateMessage("Actor", 3, "GateVImageActor -- Construct: mHalfSize of parent = " << ComputeBoundingBox(mVolume->GetLogicalVolume()->GetSolid()) << G4endl);

  if (mResolutionIsSet) {
    mVoxelSize.setX(mHalfSize.x()*2./mResolution.x());
    mVoxelSize.setY(mHalfSize.y()*2./mResolution.y());
    mVoxelSize.setZ(mHalfSize.z()*2./mResolution.z());
  }

  if (mVoxelSizeIsSet) {
    mResolution.setX(std::max(1, (int)floor(mHalfSize.x()*2/mVoxelSize.x())));
    mResolution.setY(std::max(1, (int)floor(mHalfSize.y()*2/mVoxelSize.y())));
    mResolution.setZ(std::max(1, (int)floor(mHalfSize.z()*2/mVoxelSize.z())));
  }

  mImage.SetResolutionAndVoxelSize(mResolution, mVoxelSize);

  // Update position with parent ...
  //  SetPosition(mVolume);
  //  mPositionIsSet = true;

  GateMessage("Actor", 3, "GateVImageActor -- Construct: position of parent = " <<mVolume->GetPhysicalVolume()->GetObjectTranslation()  << G4endl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct: position of frame = " <<mVolume->GetPhysicalVolume()->GetFrameTranslation()  << G4endl);

  //  mHalfSize = mImage.GetHalfSize();
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): halfsize  = " << mHalfSize << G4endl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): resol     = " << mResolution << G4endl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): voxelsize = " << mVoxelSize << G4endl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): hitType   = " << mStepHitTypeName << G4endl);

  GateDebugMessageDec("Actor", 4, "GateVImageActor -- Construct: end" << G4endl);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Sets the type of the hit (pre / post / split)
void GateVImageActor::SetStepHitType(G4String t)
{
  mStepHitTypeName = t;
  if (t == "pre")    { mStepHitType = PreStepHitType; return; }
  if (t == "post")   { mStepHitType = PostStepHitType; return; }
  if (t == "middle") { mStepHitType = MiddleStepHitType; return; }
  if (t == "random") { mStepHitType = RandomStepHitType; return; }

  GateError("GateVImageActor -- SetStepHitType: StepHitType is set to '" << t << "' while I only know 'pre', 'post', 'random' or 'middle'.");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::PreUserTrackingAction(const GateVVolume * , const G4Track * t)
{
  //GateVActor::PreUserTrackingAction(v, t);
  //int index = GetIndexFromTrackPosition(v, t);

  int index = GetIndexFromTrackPosition(GetVolume(), t);
  UserPreTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::PostUserTrackingAction(const GateVVolume * , const G4Track * t)
{
  //GateVActor::PostUserTrackingAction(v, t);
  //int index = GetIndexFromTrackPosition(v, t);

  int index = GetIndexFromTrackPosition(GetVolume(), t);
  UserPostTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::UserSteppingAction(const GateVVolume * , const G4Step * step)
{
  // GateVActor::UserSteppingAction(v, step);
  //int index = GetIndexFromStepPosition(v, step);

  int index = GetIndexFromStepPosition(GetVolume(), step);
  UserSteppingActionInVoxel(index, step);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateVImageActor::GetIndexFromTrackPosition(const GateVVolume * v , const G4Track * track)
{
  if(v==0) return -1;
  G4ThreeVector tmpPosition =  track->GetPosition();

  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(track->GetTouchable());
  int maxDepth = theTouchable->GetHistoryDepth();
  G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();
  
  GateDebugMessage("Track",3,"GateVImageActor -- GetIndexFromStepPosition: Step in "<<currentVol->GetName()<<" - Max Depth = "<<maxDepth
                                                                      <<" -> target = "<<v->GetLogicalVolume()->GetName()<<G4endl );
  int depth = 0;
  int transDepth = maxDepth;

  while((depth<maxDepth) && (currentVol != v->GetLogicalVolume()))
  {
    depth++;
    transDepth--;
    currentVol = theTouchable->GetVolume(depth)->GetLogicalVolume();
  }

  if(depth>=maxDepth) return -1;

  // GateError( "currentVol : "<< currentVol->GetName()<<"    Logical Volume "<< v->GetLogicalVolume()->GetName()<<" not found!" ); 
 
  GateDebugMessage("Step",3,"GateVImageActor -- GetIndexFromStepPosition: Logical volume "<<currentVol->GetName() <<" found! - Depth = "<<depth <<G4endl );

  G4ThreeVector position = theTouchable->GetHistory()->GetTransform(transDepth).TransformPoint(tmpPosition);


  if (mPositionIsSet) {
    GateDebugMessage("Track", 3, "GateVImageActor -- GetIndexFromStepPosition: Track position (vol reference) = " << position << G4endl);
    position.setX( position.x() - mPosition.x());
    position.setY( position.y() - mPosition.y());
    position.setZ( position.z() - mPosition.z());
  }

  GateDebugMessage("Track", 3, "GateVImageActor -- GetIndexFromStepPosition: Track position = " << position << G4endl);
  int index = mImage.GetIndexFromPosition(position);
  return index;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateVImageActor::GetIndexFromStepPosition(const GateVVolume * v, const G4Step * step)
{  
  if(v==0) return -1;

  G4ThreeVector worldPos = step->GetPostStepPoint()->GetPosition();
  G4ThreeVector worldPre =  step->GetPreStepPoint()->GetPosition() ;

  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable());
  int maxDepth = theTouchable->GetHistoryDepth();
  G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();
  
  GateDebugMessage("Step",3,"GateVImageActor -- GetIndexFromStepPosition: Step in "<<currentVol->GetName()<<" - Max Depth = "<<maxDepth
		   <<" -> target = "<<v->GetLogicalVolume()->GetName()<<G4endl );
  GateDebugMessage("Step", 3, " worldPre = " << worldPre<<G4endl);
  GateDebugMessage("Step", 3, " worldPos = " << worldPos<<G4endl);
  int depth = 0;
  int transDepth = maxDepth;
  // DD(v->GetLogicalVolume());

  while((depth<maxDepth) && 
        (currentVol->GetName() != v->GetLogicalVolume()->GetName()))
    //(currentVol != v->GetLogicalVolume())) //depth<=maxDepth or depth<maxDepth ? OK < only
    {
      depth++;
      transDepth--;
      currentVol = theTouchable->GetVolume(depth)->GetLogicalVolume();
      // DD(depth);
//       DD(transDepth);
//       DD(currentVol->GetName());
//       DD(currentVol);
    }

  if(depth>=maxDepth) return -1;

// GateError( "Logical Volume "<< v->GetLogicalVolume()->GetName()<<" not found!" ); 
  
  GateDebugMessage("Step",3,"GateVImageActor -- GetIndexFromStepPosition: Logical volume "<<currentVol->GetName() <<" found! - Depth = "<<depth <<G4endl );

  G4ThreeVector postPosition = theTouchable->GetHistory()->GetTransform(transDepth).TransformPoint(worldPos);
  G4ThreeVector prePosition = theTouchable->GetHistory()->GetTransform(transDepth).TransformPoint(worldPre);

  if (mPositionIsSet) {
    GateDebugMessage("Step", 3, "GateVImageActor -- GetIndexFromStepPosition: Step postPosition (vol reference) = " << postPosition << G4endl);
    GateDebugMessage("Step", 3, "GateVImageActor -- GetIndexFromStepPosition: Step prePosition (vol reference) = " << prePosition << G4endl);
    GateDebugMessage("Step", 3, "GateVImageActor -- GetIndexFromStepPosition: Voxel grid position = " << mPosition << G4endl);
    prePosition.setX( prePosition.x() - mPosition.x());
    prePosition.setY( prePosition.y() - mPosition.y());
    prePosition.setZ( prePosition.z() - mPosition.z());
    postPosition.setX( postPosition.x() - mPosition.x());
    postPosition.setY( postPosition.y() - mPosition.y());
    postPosition.setZ( postPosition.z() - mPosition.z());
  }

  GateDebugMessage("Step", 2, "GateVImageActor -- GetIndexFromStepPosition:Actor  UserSteppingAction (type = " << mStepHitTypeName << ")" << G4endl
		   << "\tPreStep     = " << prePosition << G4endl 
		   << "\tPostStep    = "<< postPosition << G4endl);
  
  //http://geant4-hn.slac.stanford.edu:5090/HyperNews/public/get/eventtrackmanage/263/1/1.html

  int index=-1;
  if (mStepHitType == PreStepHitType) {
    index = mImage.GetIndexFromPrePosition(prePosition, postPosition);
  }
  if (mStepHitType == PostStepHitType) {
    G4ThreeVector direction = postPosition - prePosition;
    index = mImage.GetIndexFromPostPositionAndDirection(postPosition, direction);
  }
  if (mStepHitType == MiddleStepHitType) {
    G4ThreeVector middle = prePosition + postPosition;
    middle/=2.;
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tMiddleStep  = " << middle << G4endl);
    index = mImage.GetIndexFromPosition(middle);
  }
  if (mStepHitType == RandomStepHitType) {
    G4double x = G4UniformRand();
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tx         = " << x << G4endl);
    G4ThreeVector direction = postPosition-prePosition;
    GateDebugMessageCont("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tdirection = " << direction << G4endl);
    //normalize(direction);
    //GateDebugMessageCont("Step", 4, "\tdirection = " << direction << G4endl);
    G4ThreeVector position = prePosition + x*direction;
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tRandomStep = " << position << G4endl);
    index = mImage.GetIndexFromPosition(position);
  }
  
  GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tVoxel index = " << index << G4endl);
  return index;
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEVIMAGEACTOR_CC */

