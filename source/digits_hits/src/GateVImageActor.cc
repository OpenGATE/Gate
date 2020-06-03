/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEVIMAGEACTOR_CC
#define GATEVIMAGEACTOR_CC

#include "GateVImageActor.hh"
#include "GateMiscFunctions.hh"
#include "GateObjectStore.hh"
#include "GateVImageVolume.hh"
#include "GateUtilityForG4ThreeVector.hh"

#include <G4Step.hh>
#include <G4TouchableHistory.hh>
#include <G4VoxelLimits.hh>

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
  GateMessageInc("Actor",4, "GateVImageActor() - begin\n");
  //pMessenger = new GateImageActorMessenger(this);
  GateMessageDec("Actor",4, "GateVImageActor() - end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor
GateVImageActor::~GateVImageActor()
{
  GateMessageInc("Actor",4, "~GateVImageActor() - begin\n");
  //if (pMessenger) delete pMessenger;
  GateMessageDec("Actor",4, "~GateVImageActor() - end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImageActor::ResetData()
{
  mImage.Fill(0);
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
  mHalfSize = v;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImageActor::SetSize(G4ThreeVector v)
{
  mHalfSizeIsSet = true;
  mHalfSize = v / 2.0;
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
  GateDebugMessageInc("Actor", 4, "GateVImageActor -- Construct: begin\n");
  GateVActor::Construct();

  if (!mHalfSizeIsSet){
	  if (mResolutionIsSet && mVoxelSizeIsSet){
		  mHalfSize = KroneckerProduct(mResolution, mVoxelSize)/2;
	  }
	  else {
		  mHalfSize = ComputeBoundingBox(mVolume->GetLogicalVolume()->GetSolid());
	  }
	  mHalfSizeIsSet = true;
  }
  else {
	  if (mResolutionIsSet && mVoxelSizeIsSet) {
		  GateError("GateVImageActor -- Construct: Please give a combination of two between" <<
				    " the size, the resolution and the voxelsize (not all) for the sensor");
	  }
  }

  if (!mResolutionIsSet && !mVoxelSizeIsSet) {
    mResolution = G4ThreeVector(1.0, 1.0, 1.0);
    mResolutionIsSet = true;
  }

  GateMessage("Actor", 3, "GateVImageActor -- Construct: mHalfSize of parent = " << ComputeBoundingBox(mVolume->GetLogicalVolume()->GetSolid()) << Gateendl);

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

  // Set the origin : take into account the origin of the attached volume (if exist)
  G4VoxelLimits limits;
  G4double min, max;
  G4AffineTransform origin;
  G4ThreeVector size;
  mVolume->GetLogicalVolume()->GetSolid()->CalculateExtent(kXAxis, limits, origin, min, max);
  size[0] = max-min;
  mVolume->GetLogicalVolume()->GetSolid()->CalculateExtent(kYAxis, limits, origin, min, max);
  size[1] = max-min;
  mVolume->GetLogicalVolume()->GetSolid()->CalculateExtent(kZAxis, limits, origin, min, max);
  size[2] = max-min;

  // Translation between actor's size and mothervolume's size
  mOrigin = size / 2.0 - mHalfSize;

  // Take origin into account, consider halfpixel
  mOrigin += mVolume->GetOrigin();

  // Take translation into account
  mOrigin += mPosition;
  mImage.SetOrigin(mOrigin);

  // Copy rotation matrix from attached image, if the attached volume
  // is a GateVImageVolume
  if (dynamic_cast<GateVImageVolume*>(mVolume) != 0) {
    GateVImageVolume * volAsImage = (GateVImageVolume*)mVolume;
    mImage.SetTransformMatrix(volAsImage->GetTransformMatrix());
  }

  // DEBUG
  GateMessage("Actor", 3, "GateVImageActor -- Construct: position of parent = " <<mVolume->GetPhysicalVolume()->GetObjectTranslation()  << Gateendl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct: position of frame = " <<mVolume->GetPhysicalVolume()->GetFrameTranslation()  << Gateendl);

  //  mHalfSize = mImage.GetHalfSize();
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): halfsize  = " << mHalfSize << Gateendl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): resol     = " << mResolution << Gateendl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): voxelsize = " << mVoxelSize << Gateendl);
  GateMessage("Actor", 3, "GateVImageActor -- Construct(): hitType   = " << mStepHitTypeName << Gateendl);

  GateDebugMessageDec("Actor", 4, "GateVImageActor -- Construct: end\n");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImageActor::SetOriginTransformAndFlagToImage(GateImageWithStatistic & image)
{
  // Set origin, take into account the origin of the attached volume (if exist)
  G4ThreeVector offset = mOrigin;
  image.SetOrigin(mOrigin);

  // Set transformMatrix
  image.SetTransformMatrix(mImage.GetTransformMatrix());

  // Set Overwrite flag
  image.SetOverWriteFilesFlag(mOverWriteFilesFlag);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::SetOriginTransformAndFlagToImage(GateVImage & image)
{
  // Set origin, take into account the origin of the attached volume (if exist)
  G4ThreeVector offset = mOrigin;
  image.SetOrigin(mOrigin);

  // Set transformMatrix
  image.SetTransformMatrix(mImage.GetTransformMatrix());

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
  if (t == "randomCylindricalCS") { mStepHitType = RandomStepHitTypeCylindricalCS; return;}
  if (t == "postCylindricalCS") { mStepHitType = PostStepHitTypeCylindricalCS; return;}

  GateError("GateVImageActor -- SetStepHitType: StepHitType is set to '" << t << "' while I only know 'pre', 'post', 'random' or 'middle'.");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Gets the voxel size (vc)
G4ThreeVector GateVImageActor::GetVoxelSize()
{
  return mVoxelSize;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::PreUserTrackingAction(const GateVVolume*, const G4Track * t)
{
  //GateVActor::PreUserTrackingAction(v, t);
  //int index = GetIndexFromTrackPosition(v, t);

  //assert(foo==NULL); // Probably a problem here, foo is not useful ??? Please keep this comment
  int index = GetIndexFromTrackPosition(GetVolume(), t);
  UserPreTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::PostUserTrackingAction(const GateVVolume* , const G4Track * t)
{
  //GateVActor::PostUserTrackingAction(v, t);
  //int index = GetIndexFromTrackPosition(v, t);

  //assert(foo==NULL); // Probably a problem here, foo is not useful ??? Please keep this comment
  int index = GetIndexFromTrackPosition(GetVolume(), t);
  UserPostTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImageActor::UserSteppingAction(const GateVVolume* , const G4Step * step)
{
  // GateVActor::UserSteppingAction(v, step);
  //int index = GetIndexFromStepPosition(v, step);

  //assert(foo==NULL); // Probably a problem here, foo is not useful ??? Please keep this comment
  //GetIndexFromStepPosition goes from Geant4 Coord System to 'Gate'/physical coord system.

/*TODO BRENT
if (custmframe)

else*/
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

  GateDebugMessage("Track",3,"GateVImageActor -- GetIndexFromTrackPosition: Step in "<<currentVol->GetName()<<" - Max Depth = "<<maxDepth
                                                                      <<" -> target = "<<v->GetLogicalVolume()->GetName()<< Gateendl );
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

  GateDebugMessage("Step",3,"GateVImageActor -- GetIndexFromTrackPosition: Logical volume "<<currentVol->GetName() <<" found! - Depth = "<<depth << Gateendl );

  G4ThreeVector position = theTouchable->GetHistory()->GetTransform(transDepth).TransformPoint(tmpPosition);


  if (mPositionIsSet) {
    GateDebugMessage("Track", 3, "GateVImageActor -- GetIndexFromTrackPosition: Track position (vol reference) = " << position << Gateendl);
    position -= mPosition;
  }

  GateDebugMessage("Track", 3, "GateVImageActor -- GetIndexFromTrackPosition: Track position = " << position << Gateendl);
  int index = mImage.GetIndexFromPosition(position);
  return index;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateVImageActor::GetIndexFromStepPosition(const GateVVolume * v, const G4Step * step)
{
  return GetIndexFromStepPosition2(v, step, mImage, mPositionIsSet, mPosition, mStepHitType);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateVImageActor::GetIndexFromStepPosition2(const GateVVolume * v,
                                               const G4Step * step,
                                               const GateImage & image,
                                               const bool mPositionIsSet,
                                               const G4ThreeVector mPosition,
                                               const StepHitType mStepHitType)
{
  if(v==0) return -1;

  const G4ThreeVector & worldPos = step->GetPostStepPoint()->GetPosition();
  const G4ThreeVector & worldPre =  step->GetPreStepPoint()->GetPosition() ;

  G4TouchableHistory* theTouchable = (G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable());
  int maxDepth = theTouchable->GetHistoryDepth();
  G4LogicalVolume * currentVol = theTouchable->GetVolume(0)->GetLogicalVolume();

  GateDebugMessage("Step",3,"GateVImageActor -- GetIndexFromStepPosition: Step in "<<currentVol->GetName()<<" - Max Depth = "<<maxDepth
		   <<" -> target = "<<v->GetLogicalVolume()->GetName()<< Gateendl );
  GateDebugMessage("Step", 3, " worldPre = " << worldPre<< Gateendl);
  GateDebugMessage("Step", 3, " worldPos = " << worldPos<< Gateendl);
  int depth = 0;
  int transDepth = maxDepth;

  while((depth<maxDepth) &&
        (currentVol->GetName() != v->GetLogicalVolume()->GetName()))
    //(currentVol != v->GetLogicalVolume())) //depth<=maxDepth or depth<maxDepth ? OK < only
    {
      depth++;
      transDepth--;
      currentVol = theTouchable->GetVolume(depth)->GetLogicalVolume();
    }

  if(depth>=maxDepth) return -1;

  GateDebugMessage("Step",3,"GateVImageActor -- GetIndexFromStepPosition: Logical volume "<<currentVol->GetName() <<" found! - Depth = "<<depth << Gateendl );

  G4ThreeVector postPosition = theTouchable->GetHistory()->GetTransform(transDepth).TransformPoint(worldPos);
  G4ThreeVector prePosition = theTouchable->GetHistory()->GetTransform(transDepth).TransformPoint(worldPre);

  if (mPositionIsSet) {
    GateDebugMessage("Step", 3, "GateVImageActor -- GetIndexFromStepPosition: Step postPosition (vol reference) = " << postPosition << Gateendl);
    GateDebugMessage("Step", 3, "GateVImageActor -- GetIndexFromStepPosition: Step prePosition (vol reference) = " << prePosition << Gateendl);
    GateDebugMessage("Step", 3, "GateVImageActor -- GetIndexFromStepPosition: Voxel grid position = " << mPosition << Gateendl);
    prePosition -= mPosition;
    postPosition -= mPosition;
  }

  GateDebugMessage("Step", 2, "GateVImageActor -- GetIndexFromStepPosition:Actor  UserSteppingAction (type = " << mStepHitTypeName << ")\n"
		   << "\tPreStep     = " << prePosition << Gateendl
		   << "\tPostStep    = "<< postPosition << Gateendl);

  //http://geant4-hn.slac.stanford.edu:5090/HyperNews/public/get/eventtrackmanage/263/1/1.html

  int index=-1;
  if (mStepHitType == PreStepHitType) {
    //index = mImage.GetIndexFromPrePosition(prePosition, postPosition);
    G4ThreeVector direction = postPosition - prePosition;
    index = image.GetIndexFromPostPositionAndDirection(prePosition, direction);
    //TODO Brent index = image.GetIndexFromPostPositionAndDirection(R x prePosition, direction);
  }
  if (mStepHitType == PostStepHitType) {
    G4ThreeVector direction = postPosition - prePosition;
    index = image.GetIndexFromPostPositionAndDirection(postPosition, direction);
  }
  if (mStepHitType == MiddleStepHitType) {
    G4ThreeVector middle = prePosition + postPosition;
    middle/=2.;
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tMiddleStep  = " << middle << Gateendl);
    index = image.GetIndexFromPosition(middle);
  }
  if (mStepHitType == RandomStepHitType) {
    G4double x = G4UniformRand();
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tx         = " << x << Gateendl);
    G4ThreeVector direction = postPosition-prePosition;
    GateDebugMessageCont("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tdirection = " << direction << Gateendl);
    //normalize(direction);
    //GateDebugMessageCont("Step", 4, "\tdirection = " << direction << Gateendl);
    G4ThreeVector position = prePosition + x*direction;
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tRandomStep = " << position << Gateendl);
    index = image.GetIndexFromPosition(position);
  }
 if (mStepHitType == RandomStepHitTypeCylindricalCS) {
    G4double x = G4UniformRand();
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tx         = " << x << Gateendl);
    G4ThreeVector direction = postPosition-prePosition;
    GateDebugMessageCont("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tdirection = " << direction << Gateendl);
    //normalize(direction);
    //GateDebugMessageCont("Step", 4, "\tdirection = " << direction << Gateendl);
    G4ThreeVector position = prePosition + x*direction;
    GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tRandomStep = " << position << Gateendl);
    index = image.GetIndexFromPositionCylindricalCS(position);
  }
  if (mStepHitType == PostStepHitTypeCylindricalCS) {
    index = image.GetIndexFromPositionCylindricalCS(postPosition);
  }
  GateDebugMessage("Step", 4, "GateVImageActor -- GetIndexFromStepPosition:\tVoxel index = " << index << Gateendl);
  return index;
}
//-----------------------------------------------------------------------------


#endif /* end #define GATEVIMAGEACTOR_CC */
