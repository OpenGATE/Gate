/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateHybridDoseActor : 
  \brief 
*/

#ifndef GATEHYBRIDDOSEACTOR_CC
#define GATEHYBRIDDOSEACTOR_CC

#include "GateHybridDoseActor.hh"
#include "GateHybridMultiplicityActor.hh"
#include "GateMiscFunctions.hh"
#include "GateMaterialMuHandler.hh"
#include "GateVImageVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateSourceMgr.hh"
#include "G4Run.hh"
#include "GateTrack.hh"
#include "GateActions.hh"
#include "G4Hybridino.hh"

#include <typeinfo>
//-----------------------------------------------------------------------------
GateHybridDoseActor::GateHybridDoseActor(G4String name, G4int depth) :
  GateVImageActor(name,depth) {
  mCurrentEvent=-1;
  pMessenger = new GateHybridDoseActorMessenger(this);
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
  mIsMaterialAndMuTableInitialized = false;

  // Create a 'MultiplicityActor' if not exist
  GateActorManager *actorManager = GateActorManager::GetInstance();
  G4bool noMultiplicityActor = true;
  std::vector<GateVActor*> actorList = actorManager->GetTheListOfActors();
  for(unsigned int i=0; i<actorList.size(); i++)
  {
    if(actorList[i]->GetTypeName() == "GateHybridMultiplicityActor") { noMultiplicityActor = false; }
  }
  if(noMultiplicityActor) { actorManager->AddActor("HybridMultiplicityActor","hybridMultiplicityActor"); }
  pHybridMultiplicityActor = GateHybridMultiplicityActor::GetInstance();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateHybridDoseActor::~GateHybridDoseActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

 //-----------------------------------------------------------------------------
/// Construct
void GateHybridDoseActor::Construct() {
  GateMessage("Actor", 0, " HybridDoseActor construction" << G4endl);
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
  pHybridMultiplicityActor->SetMultiplicity(mIsHybridinoEnabled, mPrimaryMultiplicity, mSecondaryMultiplicity, attachedVolume);
  mListOfRaycasting = pHybridMultiplicityActor->GetRaycastingList();

  // Get the stepping manager
  mSteppingManager = G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager();
  
  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableUserSteppingAction(true);

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
  //   GateMessage("Actor", 0," Translation " << worldToVolume.NetTranslation() << " Rotation " << worldToVolume.NetRotation() << G4endl);
  
//   GateMessage("Actor", 0, " halfSize " << mHalfSize << " resolution " << mResolution << " placement " << mPosition << G4endl);
  mResolution = dynamic_cast<GateVImageVolume*>(GetVolume())->GetImage()->GetResolution();
  mVoxelSize = dynamic_cast<GateVImageVolume*>(GetVolume())->GetImage()->GetVoxelSize();
  mHalfSize = dynamic_cast<GateVImageVolume*>(GetVolume())->GetImage()->GetHalfSize();
//   WARNING : 'Hybrid Dose Actor' inherits automatically the geometric properties of the attached volume
//   GateMessage("Actor", 0, " halfSize " << mHalfSize << " resolution " << mResolution << " voxelSize " << mVoxelSize << G4endl);
      
  // Total dose map initialisation
  mDoseFilename = G4String(removeExtension(mSaveFilename))+"-totalDose."+G4String(getExtension(mSaveFilename));
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
  
//   mDoseImage.SetOrigin(mOrigin);
//   mPrimaryDoseImage.SetOrigin(mOrigin);
//   mSecondaryDoseImage.SetOrigin(mOrigin);

  ConversionFactor = 1.60217653e-19 * 1.e6 * 1.e3 * 1.e3;
  VoxelVolume = GetDoselVolume();
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::InitializeMaterialAndMuTable()
{
  if(!mIsMaterialAndMuTableInitialized)
  {
    int lineSize = (int)lrint(mResolution.x());
    int planeSize = (int)lrint(mResolution.x()*mResolution.y());
    int voxelIndex = -1;
    
    theListOfMaterial.resize(mResolution.x()*mResolution.y()*mResolution.z());
    theListOfMuTable.resize(mResolution.x()*mResolution.y()*mResolution.z());

    GateVImageVolume* volume = dynamic_cast<GateVImageVolume*>(GetVolume());
    GateDetectorConstruction *detectorConstruction = GateDetectorConstruction::GetGateDetectorConstruction();
    for(int x=0; x<mResolution.x(); x++)
    {
      for(int y=0; y<mResolution.y(); y++)
      {
	for(int z=0; z<mResolution.z(); z++)
	{
	  voxelIndex = x+y*lineSize+z*planeSize;
	  theListOfMaterial[voxelIndex] = detectorConstruction->mMaterialDatabase.GetMaterial(volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(x,y,z)));
	  theListOfMuTable[voxelIndex] = mMaterialHandler->GetMuTable(theListOfMaterial[voxelIndex]);
	}
      }
    }
    
    mIsMaterialAndMuTableInitialized = true;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateHybridDoseActor::SaveData()
{
  if(mIsDoseImageEnabled) { mDoseImage.SaveData(mCurrentEvent+1, false); }
  if(mIsPrimaryDoseImageEnabled) { mPrimaryDoseImage.SaveData(mCurrentEvent+1, false); }
  if(mIsSecondaryDoseImageEnabled) { mSecondaryDoseImage.SaveData(mCurrentEvent+1, false); }

  if(mIsLastHitEventImageEnabled) { mLastHitEventImage.Fill(-1); /* reset */ }
  if(mIsPrimaryLastHitEventImageEnabled) { mPrimaryLastHitEventImage.Fill(-1); /* reset */ }
  if(mIsSecondaryLastHitEventImageEnabled) { mSecondaryLastHitEventImage.Fill(-1); /* reset */ }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::ResetData()
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
void GateHybridDoseActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateHybridDoseActor -- Begin of Run" << G4endl);
  // ResetData(); // Do no reset here !! (when multiple run);

  // security on attachedVolume
  GateVImageVolume* volume = dynamic_cast<GateVImageVolume*>(GetVolume());
  if(!volume) { GateError("Error in " << GetName() << ": GateVImageVolume doesn't exist"); }
  
  // fast material and mu access
  if(!mIsMaterialAndMuTableInitialized) { InitializeMaterialAndMuTable(); }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateHybridDoseActor::BeginOfEventAction(const G4Event *) { 
//   GateVActor::BeginOfEventAction(event);
  mCurrentEvent++;
  if(mCurrentEvent % 100 == 0)
  {
    GateMessage("Actor", 0, " BeginOfEventAction - " << mCurrentEvent << " - primary weight = " << 1./mPrimaryMultiplicity << " - secondary weight = " << 1./mSecondaryMultiplicity << G4endl);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::PreUserTrackingAction(const GateVVolume *, const G4Track *) {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::PostUserTrackingAction(const GateVVolume *, const G4Track *) {}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//G4bool GateDoseActor::ProcessHits(G4Step * step , G4TouchableHistory* th)
void GateHybridDoseActor::UserSteppingAction(const GateVVolume *v, const G4Step* step)
{
//   DD("Dose step");
  
  GateVImageActor::UserSteppingAction(v, step);
  
  if(step->GetTrack()->GetDefinition()->GetParticleName() == "hybridino"){  
    RayCast(step);
//     GateMessage("Actor", 0, step->GetTrack()->GetCurrentStepNumber() << " Step weight = " << step->GetTrack()->GetWeight() << G4endl);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::UserSteppingActionInVoxel(const int /*index*/, const G4Step *) {}
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
void GateHybridDoseActor::RayCast(const G4Step* step)
{
//   GateMessage("Actor", 0, "  " << G4endl;);
//   GateMessage("Actor", 0, " halfSize " << mHalfSize << " resolution " << mResolution << " voxelSize " << mVoxelSize << G4endl);

  G4StepPoint *preStep(step->GetPreStepPoint());
  G4ThreeVector position = preStep->GetPosition();
  G4ThreeVector momentum = preStep->GetMomentumDirection();
  G4double energy = preStep->GetKineticEnergy();

//   GateMessage("Actor", 0, " before : position " << position << " momentum " << momentum << G4endl);
  position = worldToVolume.TransformPoint(position);  
  momentum.transform(mRotationMatrix);
//   GateMessage("Actor", 0, " after  : position " << position << " momentum " << momentum << G4endl);
  
  int xincr = getIncrement(momentum.x());
  int yincr = getIncrement(momentum.y());
  int zincr = getIncrement(momentum.z());
  
  //Voxels size
  double dx = 2.0 * mHalfSize.x() / mResolution.x();
  double dy = 2.0 * mHalfSize.y() / mResolution.y();
  double dz = 2.0 * mHalfSize.z() / mResolution.z();

  //Momentum norm have to be 1
  //double norm = momentum.mag();
  double Lx = xincr * dx / momentum.x(); //*norm
  double Ly = yincr * dy / momentum.y(); //*norm
  double Lz = zincr * dz / momentum.z(); //*norm
  //Coordinates
  int x, y, z;
  //Because GateImage is shit
  //mImage.GetCoordinatesFromPosition(position, x, y, z);
  double xtemp = (position.x()+mHalfSize.x())/dx;
  double ytemp = (position.y()+mHalfSize.y())/dy;
  double ztemp = (position.z()+mHalfSize.z())/dz;
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
  double Rx, Ry, Rz;
  double Ltot = 2.0*mHalfSize.x()*mHalfSize.y()*mHalfSize.z();
  double Ltmp = Ltot;

//   GateMessage("Actor", 0, "increment " << xincr << " " << yincr << " " << zincr << G4endl);
  
  if(xincr > 0){
    Rx = xincr * (-mHalfSize.x() + (x+1)*dx - position.x())/momentum.x();
    Ltot = (mHalfSize.x() - position.x())/momentum.x();
  }
  else if (momentum.x() < 0){
    Rx = xincr * (position.x() - (-mHalfSize.x() + x*dx))/momentum.x();
    Ltot = (-mHalfSize.x() - position.x())/momentum.x();
  }
  else {Rx = 2*Ltot;}
      
  if(yincr > 0){
      Ry = yincr * (-mHalfSize.y() + (y+1)*dy - position.y())/momentum.y();
      Ltmp = (mHalfSize.y() - position.y())/momentum.y();
  }
  else if(momentum.y() < 0){
    Ry = yincr * (position.y() - (-mHalfSize.y() + y*dy))/momentum.y();
    Ltmp = (-mHalfSize.y() - position.y())/momentum.y();
  }
  else {Ry = 2*Ltot;}
  if (Ltmp < Ltot) {Ltot = Ltmp;}
  
  if(zincr > 0){
    Rz = zincr * (-mHalfSize.z() + (z+1)*dz - position.z())/momentum.z();
      Ltmp = (mHalfSize.z() - position.z())/momentum.z();
  }
  else if (momentum.z() < 0)  {
      Rz = zincr * (position.z() - (-mHalfSize.z() + z*dz))/momentum.z();
      Ltmp = (-mHalfSize.z() - position.z())/momentum.z();
  }
  else {Rz = 2*Ltot;}
    
  if (Ltmp < Ltot) {Ltot = Ltmp;}
  
//   GateMessage("Actor", 0, "Lx : " << Lx << " Ly " << Ly << " Lz " << Lz << G4endl);
//   GateMessage("Actor", 0, "Rx : " << Rx << " Ry " << Ry << " Rz " << Rz << G4endl);
//   GateMessage("Actor", 0, "entry index " << x << " " << y << " " << z << G4endl);
//   GateMessage("Actor", 0, "LTot" << Ltot << G4endl); 
  
  int lineSize = (int)lrint(mResolution.x());
  int planeSize = (int)lrint(mResolution.x()*mResolution.y());
  int index = 0;
  G4Material *material;
  double dose = 0.0;
  double L = 0.0;
  
  double hybridTrackWeight = pHybridMultiplicityActor->GetHybridTrackWeight();
  double delta_in  = hybridTrackWeight;
  double delta_out(0.0);
  double mu(0.0);
  double muen(0.0);

  // test on dose contribution: primary or secondary ?
  GateImageWithStatistic *currentDoseImage = 0;
  GateImage *currentLastHitImage = 0;
  bool isCurrentLastHitImageEnabled = false;
  bool isCurrentDoseUncertaintyEnabled = false;
  if(step->GetTrack()->GetParentID() == 0)
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

  while(L < Ltot-0.00001)
  {
//     GateMessage("Actor", 0, " index " << x << " " << y << " " << z << " | Rest " << Rx << " " << Ry << " " << Rz << " | L " << L << " Ltot " << Ltot << G4endl);
//     GateMessage("Actor", 0, " Rest  " << Rx << " " << Ry << " " << Rz << G4endl);
//     GateMessage("Actor", 0, "L " << L << " Ltot " << Ltot << G4endl);
//     GateMessage("Actor", 0, "Volume = " << GetVolume()->GetLogicalVolumeName() << G4endl);
//     GateMessage("Actor", 0, "material 1 : " << volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(x,y,z)) << G4endl);    
//     GateMessage("Actor", 0, "label : " << volume->GetImage()->GetValue(0,0,0) << " " << volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(0,0,0)) << G4endl);    

    index = x+y*lineSize+z*planeSize;

    bool sameEvent = true;
    if(mIsLastHitEventImageEnabled)
    {
      GateDebugMessage("Actor", 2,  "GateHybridDoseActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
      if(mCurrentEvent != mLastHitEventImage.GetValue(index))
      {
	sameEvent = false;
	mLastHitEventImage.SetValue(index, mCurrentEvent);
      }
    }
    
    bool currentContributionSameEvent = true;
    if(isCurrentLastHitImageEnabled)
    {
      GateDebugMessage("Actor", 2,  "GateHybridDoseActor -- UserSteppingActionInVoxel: Last event in index = " << currentLastHitImage->GetValue(index) << G4endl);
      if(mCurrentEvent != currentLastHitImage->GetValue(index))
      {
	currentContributionSameEvent = false;
	currentLastHitImage->SetValue(index, mCurrentEvent);
      }
    }
    
    material = theListOfMaterial[index];
    mu = theListOfMuTable[index]->GetMu(energy)*material->GetDensity()/(g/cm3);
    muen = theListOfMuTable[index]->GetMuEn(energy);

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
    
    dose = ConversionFactor*energy*muen*(delta_in-delta_out)/mu/VoxelVolume;

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
    
//     if(mIsDoseImageEnabled) { mDoseImage.AddValue(index, doseValue); }
//     if(currentDoseImage) { currentDoseImage->AddValue(index, doseValue); }
    
    delta_in = delta_out;
  }

  // New initialisation of the track to overcome the navigation process
  G4ThreeVector newMomentum = preStep->GetMomentumDirection();
  G4ThreeVector newPosition = preStep->GetPosition();
  G4double newTrackLength = Ltot + 0.00001; // add a small distance to directly begin outside the voxelised volume
  newPosition.set(newPosition.x() + newTrackLength*newMomentum.x(), newPosition.y() + newTrackLength*newMomentum.y(), newPosition.z() + newTrackLength*newMomentum.z());
  G4Track *modifiedTrack = step->GetTrack();
  modifiedTrack->SetPosition(newPosition);
  mSteppingManager->SetInitialStep(modifiedTrack);
  
  // register the new track and corresponding weight into the trackList (see 'HybridMultiplicityActor')
  pHybridMultiplicityActor->SetHybridTrackWeight(delta_in);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEHYBRIDDOSEACTOR_CC */

