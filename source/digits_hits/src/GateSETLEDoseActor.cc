/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*
  \brief Class GateSETLEDoseActor :
  \brief
*/

#ifndef GATESETLEDOSEACTOR_CC
#define GATESETLEDOSEACTOR_CC

#include "GateSETLEDoseActor.hh"
#include "GateSETLEMultiplicityActor.hh"
#include "GateMiscFunctions.hh"
#include "GateMaterialMuHandler.hh"
#include "GateVImageVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateSourceMgr.hh"
#include "G4Run.hh"
#include "GateTrack.hh"
#include "GateActions.hh"
#include "G4Hybridino.hh"
#include "G4RegionStore.hh"
#include "G4PhysicalConstants.hh"

#include <typeinfo>
//-----------------------------------------------------------------------------
GateSETLEDoseActor::GateSETLEDoseActor(G4String name, G4int depth) :
  GateVImageActor(name,depth) {
  mCurrentEvent=-1;
  pMessenger = new GateSETLEDoseActorMessenger(this);
  mMaterialHandler = GateMaterialMuHandler::GetInstance();
  mListOfRaycasting = 0;

  mIsDoseImageEnabled = true;
  mIsDoseUncertaintyImageEnabled = false;
  mIsPrimaryDoseImageEnabled = false;
  mIsPrimaryDoseUncertaintyImageEnabled = false;
  mIsSecondaryDoseImageEnabled = false;
  mIsSecondaryDoseUncertaintyImageEnabled = false;

  mIsLastHitEventImageEnabled = false;
  mIsPrimaryLastHitEventImageEnabled = false;
  mIsSecondaryLastHitEventImageEnabled = false;

  mIsHybridinoEnabled = false;
  mIsMuTableInitialized = false;

  // Create a 'MultiplicityActor' if not exist
  GateActorManager *actorManager = GateActorManager::GetInstance();
  G4bool noMultiplicityActor = true;
  std::vector<GateVActor*> actorList = actorManager->GetTheListOfActors();
  for(unsigned int i=0; i<actorList.size(); i++)
    {
      if(actorList[i]->GetTypeName() == "GateSETLEMultiplicityActor") { noMultiplicityActor = false; }
    }
  if(noMultiplicityActor) { actorManager->AddActor("SETLEMultiplicityActor","seTLEMultiplicityActor"); }
  pSETLEMultiplicityActor = GateSETLEMultiplicityActor::GetInstance();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateSETLEDoseActor::~GateSETLEDoseActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateSETLEDoseActor::Construct() {
  GateMessage("Actor", 0, " SETLEDoseActor construction\n");
  GateVImageActor::Construct();

  // Multiplicity initialisation
  // --> Find the couple ('physicalG4Volume','secondaryMultiplicity') in 'MultiplicityActor'
  // WARNING only works with voxelized volume
  G4VPhysicalVolume *attachedVolume = GetVolume()->GetPhysicalVolume();
  int daughterNumber = attachedVolume->GetLogicalVolume()->GetNoDaughters();
  while(daughterNumber)
    {
      attachedVolume = attachedVolume->GetLogicalVolume()->GetDaughter(0);
      daughterNumber = attachedVolume->GetLogicalVolume()->GetNoDaughters();
    }
  // --> Set primary and secondary multiplicities in 'MultiplicityActor'
  pSETLEMultiplicityActor->SetMultiplicity(mIsHybridinoEnabled, mPrimaryMultiplicity, mSecondaryMultiplicity, attachedVolume);
  mListOfRaycasting = pSETLEMultiplicityActor->GetRaycastingList();

  // Get the stepping manager
  mSteppingManager = G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableUserSteppingAction(true);

  mResolution = dynamic_cast<GateVImageVolume*>(GetVolume())->GetImage()->GetResolution();
  mVoxelSize = dynamic_cast<GateVImageVolume*>(GetVolume())->GetImage()->GetVoxelSize();
  mHalfSize = dynamic_cast<GateVImageVolume*>(GetVolume())->GetImage()->GetHalfSize();
  // 'SETLE Dose Actor' inherits automatically the geometric properties of the attached volume

  // Total dose map initialisation
  mDoseFilename = G4String(removeExtension(mSaveFilename))+"-Dose."+G4String(getExtension(mSaveFilename));
  if (mIsDoseImageEnabled)
    {
      mDoseImage.EnableSquaredImage(false);
      mDoseImage.EnableUncertaintyImage(mIsDoseUncertaintyImageEnabled);
      mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      // Force the computation of squared image if uncertainty is enabled
      if(mIsDoseUncertaintyImageEnabled)
        {
          mDoseImage.EnableSquaredImage(true);
          mIsLastHitEventImageEnabled = true;
          mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
          mLastHitEventImage.Allocate();
        }
      mDoseImage.Allocate();
      mDoseImage.SetFilename(mDoseFilename);
    }

  // Primary dose map initialisation
  mPrimaryDoseFilename = G4String(removeExtension(mSaveFilename))+"-primaryDose."+G4String(getExtension(mSaveFilename));
  if (mIsPrimaryDoseImageEnabled)
    {
      mPrimaryDoseImage.EnableSquaredImage(false);
      mPrimaryDoseImage.EnableUncertaintyImage(mIsPrimaryDoseUncertaintyImageEnabled);
      mPrimaryDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      // Force the computation of squared image if uncertainty is enabled
      if(mIsPrimaryDoseUncertaintyImageEnabled)
        {
          mPrimaryDoseImage.EnableSquaredImage(true);
          mIsPrimaryLastHitEventImageEnabled = true;
          mPrimaryLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
          mPrimaryLastHitEventImage.Allocate();
        }
      mPrimaryDoseImage.Allocate();
      mPrimaryDoseImage.SetFilename(mPrimaryDoseFilename);
    }

  // Secondary dose map initialisation
  mSecondaryDoseFilename = G4String(removeExtension(mSaveFilename))+"-secondaryDose."+G4String(getExtension(mSaveFilename));
  if (mIsSecondaryDoseImageEnabled)
    {
      mSecondaryDoseImage.EnableSquaredImage(false);
      mSecondaryDoseImage.EnableUncertaintyImage(mIsSecondaryDoseUncertaintyImageEnabled);
      mSecondaryDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      // Force the computation of squared image if uncertainty is enabled
      if(mIsSecondaryDoseUncertaintyImageEnabled)
        {
          mSecondaryDoseImage.EnableSquaredImage(true);
          mIsSecondaryLastHitEventImageEnabled = true;
          mSecondaryLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
          mSecondaryLastHitEventImage.Allocate();
        }
      mSecondaryDoseImage.Allocate();
      mSecondaryDoseImage.SetFilename(mSecondaryDoseFilename);
    }

  // Initialize raycasting members
  mBoxMin[0] = -mHalfSize.x();
  mBoxMin[1] = -mHalfSize.y();
  mBoxMin[2] = -mHalfSize.z();
  mBoxMax[0] = mHalfSize.x();
  mBoxMax[1] = mHalfSize.y();
  mBoxMax[2] = mHalfSize.z();
  mLineSize = (int)lrint(mResolution.x());
  mPlaneSize = (int)lrint(mResolution.x()*mResolution.y());

  ConversionFactor = e_SI * 1.0e12;
  VoxelVolume = GetDoselVolume();
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEDoseActor::InitializeMaterialAndMuTable()
{
  if(!mIsMuTableInitialized)
    {
      int lineSize = (int)lrint(mResolution.x());
      int planeSize = (int)lrint(mResolution.x()*mResolution.y());
      int voxelIndex = -1;

      mListOfMuTable.resize(mResolution.x()*mResolution.y()*mResolution.z());

      GateVImageVolume* volume = dynamic_cast<GateVImageVolume*>(GetVolume());
      G4Region *region = G4RegionStore::GetInstance()->GetRegion(volume->GetObjectName());
      GateDetectorConstruction *detectorConstruction = GateDetectorConstruction::GetGateDetectorConstruction();
      for(int x=0; x<mResolution.x(); x++)
        {
          for(int y=0; y<mResolution.y(); y++)
            {
              for(int z=0; z<mResolution.z(); z++)
                {
                  voxelIndex = x+y*lineSize+z*planeSize;
                  G4Material *material = detectorConstruction->mMaterialDatabase.GetMaterial(volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(x,y,z)));
                  mListOfMuTable[voxelIndex] = mMaterialHandler->GetMuTable(region->FindCouple(material));
                }
            }
        }

      mIsMuTableInitialized = true;
    }

  // Get G4Material of the world for exponential attenuation
  GateVVolume * v = GetVolume();
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
  }
  mWorldCouple = v->GetLogicalVolume()->GetMaterialCutsCouple();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateSETLEDoseActor::SaveData()
{
  DD(mOrigin);

  if(mIsDoseImageEnabled) {
    SetOriginTransformAndFlagToImage(mDoseImage);
    mDoseImage.SaveData(mCurrentEvent+1, false);
  }
  if(mIsPrimaryDoseImageEnabled) {
    SetOriginTransformAndFlagToImage(mPrimaryDoseImage);
    mPrimaryDoseImage.SaveData(mCurrentEvent+1, false);
  }
  if(mIsSecondaryDoseImageEnabled) {
    SetOriginTransformAndFlagToImage(mSecondaryDoseImage);
    mSecondaryDoseImage.SaveData(mCurrentEvent+1, false);
  }

  // Reset
 if(mIsLastHitEventImageEnabled) mLastHitEventImage.Fill(-1);
 if(mIsPrimaryLastHitEventImageEnabled) mPrimaryLastHitEventImage.Fill(-1);
 if(mIsSecondaryLastHitEventImageEnabled) mSecondaryLastHitEventImage.Fill(-1);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEDoseActor::ResetData()
{
  if(mIsLastHitEventImageEnabled) { mLastHitEventImage.Fill(-1); }
  if(mIsPrimaryLastHitEventImageEnabled) { mPrimaryLastHitEventImage.Fill(-1); }
  if(mIsSecondaryLastHitEventImageEnabled) { mSecondaryLastHitEventImage.Fill(-1); }

  if(mIsDoseImageEnabled) { mDoseImage.Reset(); }
  if(mIsPrimaryDoseImageEnabled) { mPrimaryDoseImage.Reset(); }
  if(mIsSecondaryDoseImageEnabled) { mSecondaryDoseImage.Reset(); }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEDoseActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateSETLEDoseActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);

  // security on attachedVolume
  GateVImageVolume* volume = dynamic_cast<GateVImageVolume*>(GetVolume());
  if(!volume) { GateError("Error in " << GetName() << ": GateVImageVolume doesn't exist"); }

  // fast material and mu access
  if(!mIsMuTableInitialized) { InitializeMaterialAndMuTable(); }

  // Affine transform and rotation matrix for Raycasting
  GateVVolume * v = GetVolume();
  G4VPhysicalVolume * phys = v->GetPhysicalVolume();
  G4AffineTransform volumeToWorld = G4AffineTransform(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    volumeToWorld = volumeToWorld * x;
  }

  mRotationMatrix = volumeToWorld.NetRotation();
  worldToVolume = volumeToWorld.Inverse();
  //   GateMessage("Actor", 0," Translation " << worldToVolume.NetTranslation() << " Rotation " << worldToVolume.NetRotation() << Gateendl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateSETLEDoseActor::BeginOfEventAction(const G4Event *) {
  //   GateVActor::BeginOfEventAction(event);
  mCurrentEvent++;
  // if(mCurrentEvent % 100 == 0)
  // {
  //   GateMessage("Actor", 0, " BeginOfEventAction - " << mCurrentEvent << " - primary weight = " << 1./mPrimaryMultiplicity << " - secondary weight = " << 1./mSecondaryMultiplicity << Gateendl);
  // }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEDoseActor::PreUserTrackingAction(const GateVVolume *, const G4Track *) {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEDoseActor::PostUserTrackingAction(const GateVVolume *, const G4Track *) {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//G4bool GateDoseActor::ProcessHits(G4Step * step , G4TouchableHistory* th)
void GateSETLEDoseActor::UserSteppingAction(const GateVVolume *v, const G4Step* step)
{
  GateVImageActor::UserSteppingAction(v, step);

  if(!mIsHybridinoEnabled)
    {
      for(unsigned int r=0; r<mListOfRaycasting->size(); r++)
        {
          bool isPrimary = (*mListOfRaycasting)[r].isPrimary;
          double energy  = (*mListOfRaycasting)[r].energy;
          double weight  = (*mListOfRaycasting)[r].weight;
          G4ThreeVector position = (*mListOfRaycasting)[r].position;
          G4ThreeVector momentum = (*mListOfRaycasting)[r].momentum;

          position = worldToVolume.TransformPoint(position);
          momentum.transform(mRotationMatrix);

          bool interceptBox = IntersectionBox(position, momentum);
          if(interceptBox)
            {
              if(mNearestDistance > 0.0)
                {
                  double muWorld = mMaterialHandler->GetMu(mWorldCouple, energy);
                  weight = weight * exp(-muWorld * mNearestDistance / 10.);
                  position = position + (mNearestDistance * momentum);
                  mFarthestDistance = mFarthestDistance - mNearestDistance;
                  mNearestDistance = 0.0;
                }

              weight = RayCast(isPrimary, energy, weight, position, momentum);
            }
        }

      mListOfRaycasting->clear();
    }
  else if(step->GetTrack()->GetDefinition()->GetParticleName() == "hybridino")
    {
      bool isPrimary = false;
      if(step->GetTrack()->GetParentID() == 0) { isPrimary = true; }

      G4StepPoint *preStep(step->GetPreStepPoint());
      double energy = preStep->GetKineticEnergy();
      double weight = pSETLEMultiplicityActor->GetHybridTrackWeight();
      G4ThreeVector position = preStep->GetPosition();
      G4ThreeVector momentum = preStep->GetMomentumDirection();

      position = worldToVolume.TransformPoint(position);
      momentum.transform(mRotationMatrix);

      bool interceptBox = IntersectionBox(position, momentum);
      if(!interceptBox) { GateError("Error in GateSETLEDoseActor: intercept box failed"); }

      weight = RayCast(isPrimary, energy, weight, position, momentum);

      // New initialisation of the track
      G4ThreeVector newMomentum = preStep->GetMomentumDirection();
      G4ThreeVector newPosition = preStep->GetPosition();
      G4double newTrackLength = mTotalLength + 0.00001; // add a small distance to directly begin outside the voxelised volume

      newPosition = newPosition + newTrackLength * newMomentum;
      G4Track *modifiedTrack = step->GetTrack();
      modifiedTrack->SetPosition(newPosition);
      mSteppingManager->SetInitialStep(modifiedTrack);

      // register the new track and corresponding weight into the trackList (see 'SETLEMultiplicityActor')
      pSETLEMultiplicityActor->SetHybridTrackWeight(weight);
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSETLEDoseActor::UserSteppingActionInVoxel(const int /*index*/, const G4Step *) {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
static inline int getIncrement(double value)
{
  if (value > 0.0)
    return 1;
  return -1;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateSETLEDoseActor::RayCast(bool isPrimary, double energy, double weight, G4ThreeVector position, G4ThreeVector momentum)
{
  //   GateMessage("Actor", 0, "  \n";);
  //   GateMessage("Actor", 0, " halfSize " << mHalfSize << " resolution " << mResolution << " voxelSize " << mVoxSize << Gateendl);

  int xincr = getIncrement(momentum.x());
  int yincr = getIncrement(momentum.y());
  int zincr = getIncrement(momentum.z());

  //Momentum norm have to be 1
  //double norm = momentum.mag();
  double Lx = xincr * mVoxelSize.x() / momentum.x(); //*norm
  double Ly = yincr * mVoxelSize.y() / momentum.y(); //*norm
  double Lz = zincr * mVoxelSize.z() / momentum.z(); //*norm
  //Coordinates
  int x, y, z;
  //Because GateImage is shit
  //mImage.GetCoordinatesFromPosition(position, x, y, z);
  double xtemp = (position.x()+mHalfSize.x())/mVoxelSize.x();
  double ytemp = (position.y()+mHalfSize.y())/mVoxelSize.y();
  double ztemp = (position.z()+mHalfSize.z())/mVoxelSize.z();
  x = (int) floor(xtemp);
  y = (int) floor(ytemp);
  z = (int) floor(ztemp);

  //Security on matrix boundaries
  if(x < 0) { x = 0; }
  else if(x >= mResolution.x()) { x = mResolution.x()-1; }

  if(y < 0) { y = 0; }
  else if(y >= mResolution.y()) { y = mResolution.y()-1; }

  if(z < 0) { z = 0; }
  else if(z >= mResolution.z()) { z = mResolution.z()-1; }

  //Remaining and Total lengths
  double Rx = 1.0e15;
  double Ry = 1.0e15;
  double Rz = 1.0e15;

  mTotalLength = 2.0*mHalfSize.x()*mHalfSize.y()*mHalfSize.z();
  double Ltmp = mTotalLength;

  //   GateMessage("Actor", 0, "increment " << xincr << " " << yincr << " " << zincr << Gateendl);

  if(xincr > 0){
    Rx = xincr * (-mHalfSize.x() + (x+1)*mVoxelSize.x() - position.x())/momentum.x();
    mTotalLength = (mHalfSize.x() - position.x())/momentum.x();
  }
  else if (momentum.x() < 0){
    Rx = xincr * (position.x() - (-mHalfSize.x() + x*mVoxelSize.x()))/momentum.x();
    mTotalLength = (-mHalfSize.x() - position.x())/momentum.x();
  }
  else {Rx = 2*mTotalLength;}

  if(yincr > 0){
    Ry = yincr * (-mHalfSize.y() + (y+1)*mVoxelSize.y() - position.y())/momentum.y();
    Ltmp = (mHalfSize.y() - position.y())/momentum.y();
  }
  else if(momentum.y() < 0){
    Ry = yincr * (position.y() - (-mHalfSize.y() + y*mVoxelSize.y()))/momentum.y();
    Ltmp = (-mHalfSize.y() - position.y())/momentum.y();
  }
  else {Ry = 2*mTotalLength;}
  if (Ltmp < mTotalLength) {mTotalLength = Ltmp;}

  if(zincr > 0){
    Rz = zincr * (-mHalfSize.z() + (z+1)*mVoxelSize.z() - position.z())/momentum.z();
    Ltmp = (mHalfSize.z() - position.z())/momentum.z();
  }
  else if (momentum.z() < 0)  {
    Rz = zincr * (position.z() - (-mHalfSize.z() + z*mVoxelSize.z()))/momentum.z();
    Ltmp = (-mHalfSize.z() - position.z())/momentum.z();
  }
  else {Rz = 2*mTotalLength;}

  if (Ltmp < mTotalLength) {mTotalLength = Ltmp;}

  //   GateMessage("Actor", 0, "Lx : " << Lx << " Ly " << Ly << " Lz " << Lz << Gateendl);
  //   GateMessage("Actor", 0, "Rx : " << Rx << " Ry " << Ry << " Rz " << Rz << Gateendl);
  //   GateMessage("Actor", 0, "entry index " << x << " " << y << " " << z << Gateendl);
  //   GateMessage("Actor", 0, "LTot" << mTotalLength << Gateendl);

  int index = 0;
  double dose = 0.0;
  double L = 0.0;

  double delta_in  = weight;
  double delta_out(0.0);
  double mu(0.0);
  double muenOverRho(0.0);

  // test on dose contribution: primary or secondary ?
  GateImageWithStatistic *currentDoseImage = 0;
  GateImage *currentLastHitImage = 0;
  bool isCurrentLastHitImageEnabled = false;
  bool isCurrentDoseUncertaintyEnabled = false;
  if(isPrimary)
    {
      if(mIsPrimaryDoseImageEnabled)
        {
          currentDoseImage = &mPrimaryDoseImage;
          isCurrentLastHitImageEnabled = mIsPrimaryLastHitEventImageEnabled;
          isCurrentDoseUncertaintyEnabled = mIsPrimaryDoseUncertaintyImageEnabled;
          currentLastHitImage = &mPrimaryLastHitEventImage;
        }
    }
  else if(mIsSecondaryDoseImageEnabled)
    {
      currentDoseImage = &mSecondaryDoseImage;
      isCurrentLastHitImageEnabled = mIsSecondaryLastHitEventImageEnabled;
      isCurrentDoseUncertaintyEnabled = mIsSecondaryDoseUncertaintyImageEnabled;
      currentLastHitImage = &mSecondaryLastHitEventImage;
    }

  while(L < mTotalLength-0.00001)
    {
      //     GateMessage("Actor", 0, " index " << x << " " << y << " " << z << " | Rest " << Rx << " " << Ry << " " << Rz << " | L " << L << " mTotalLength " << mTotalLength << Gateendl);
      //     GateMessage("Actor", 0, " Rest  " << Rx << " " << Ry << " " << Rz << Gateendl);
      //     GateMessage("Actor", 0, "L " << L << " mTotalLength " << mTotalLength << Gateendl);
      //     GateMessage("Actor", 0, "Volume = " << GetVolume()->GetLogicalVolumeName() << Gateendl);
      //     GateMessage("Actor", 0, "material 1 : " << volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(x,y,z)) << Gateendl);
      //     GateMessage("Actor", 0, "label : " << volume->GetImage()->GetValue(0,0,0) << " " << volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(0,0,0)) << Gateendl);

      index = x+y*mLineSize+z*mPlaneSize;

      bool sameEvent = true;
      if(mIsLastHitEventImageEnabled)
        {
          GateDebugMessage("Actor", 2,  "GateSETLEDoseActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << Gateendl);
          if(mCurrentEvent != mLastHitEventImage.GetValue(index))
            {
              sameEvent = false;
              mLastHitEventImage.SetValue(index, mCurrentEvent);
            }
        }

      bool currentContributionSameEvent = true;
      if(isCurrentLastHitImageEnabled)
        {
          GateDebugMessage("Actor", 2,  "GateSETLEDoseActor -- UserSteppingActionInVoxel: Last event in index = " << currentLastHitImage->GetValue(index) << Gateendl);
          if(mCurrentEvent != currentLastHitImage->GetValue(index))
            {
              currentContributionSameEvent = false;
              currentLastHitImage->SetValue(index, mCurrentEvent);
            }
        }

      mu = mListOfMuTable[index]->GetMu(energy);
      muenOverRho = mListOfMuTable[index]->GetMuEnOverRho(energy);

      if(Rx < Ry && Rx < Rz){
        delta_out = delta_in*exp(-mu*Rx/10.);
        L+=Rx;
        Ry-=Rx;
        Rz-=Rx;
        Rx=Lx;
        x+=xincr;
      }
      else if (Ry < Rz){
        delta_out = delta_in*exp(-mu*Ry/10.);
        L+=Ry;
        Rx-=Ry;
        Rz-=Ry;
        Ry=Ly;
        y+=yincr;
      }
      else{
        delta_out = delta_in*exp(-mu*Rz/10.);
        L+=Rz;
        Rx-=Rz;
        Ry-=Rz;
        Rz=Lz;
        z+=zincr;
      }

      dose = ConversionFactor*energy*muenOverRho*(delta_in-delta_out)/mu/VoxelVolume;

      if(mIsDoseImageEnabled)
        {
          if(mIsDoseUncertaintyImageEnabled)
            {
              if(sameEvent) { mDoseImage.AddTempValue(index, dose); }
              else { mDoseImage.AddValueAndUpdate(index, dose); }
            }
          else { mDoseImage.AddValue(index, dose); }
        }

      if(currentDoseImage)
        {
          if(isCurrentDoseUncertaintyEnabled)
            {
              if(currentContributionSameEvent) { currentDoseImage->AddTempValue(index, dose); }
              else { currentDoseImage->AddValueAndUpdate(index, dose); }
            }
          else { currentDoseImage->AddValue(index, dose); }
        }

      delta_in = delta_out;
    }

  return delta_out;
}
//-----------------------------------------------------------------------------

bool GateSETLEDoseActor::IntersectionBox(G4ThreeVector p, G4ThreeVector m)
{
  //   double rayOrigin[3];
  mRayOrigin[0] = p.x();
  mRayOrigin[1] = p.y();
  mRayOrigin[2] = p.z();
  //   double rayDirection[3];
  mRayDirection[0] = m.x();
  mRayDirection[1] = m.y();
  mRayDirection[2] = m.z();

  mNearestDistance = -1.0e15;
  mFarthestDistance = 1.0e15;

  double T1,T2,tmpT(0.0);

  for(int i=0; i<3; i++)
    {
      if(mRayDirection[i] == 0.0) {
        if(mRayOrigin[i]<mBoxMin[i] || mRayOrigin[i]>mBoxMax[i]) { return false; }
      }

      T1 = (mBoxMin[i] - mRayOrigin[i]) / mRayDirection[i];
      T2 = (mBoxMax[i] - mRayOrigin[i]) / mRayDirection[i];

      if(T1 > T2) {
        tmpT = T1;
        T1 = T2;
        T2 = tmpT;
      }
      if(T1 > mNearestDistance) { mNearestDistance = T1; }
      if(T2 < mFarthestDistance) { mFarthestDistance = T2; }
      if(mNearestDistance > mFarthestDistance) { return false; }
      if(mFarthestDistance < 0.0) { return false; }
    }

  //   G4ThreeVector nearestPoint = pos + (nearestDistance * mmt);
  //   G4ThreeVector farthestPoint = pos + (farthestDistance * mmt);

  return true;
}

//-----------------------------------------------------------------------------

#endif /* end #define GATEHYBRIDDOSEACTOR_CC */
