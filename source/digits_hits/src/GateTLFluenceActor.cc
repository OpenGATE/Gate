/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateTLFluenceActor :
  \brief
*/

#ifndef GATETLEDOSEACTOR_CC 
#define GATETLEDOSEACTOR_CC

#include "GateTLFluenceActor.hh"
#include "GateMiscFunctions.hh"
#include "GateMaterialMuHandler.hh"
#include <G4PhysicalConstants.hh>

//-----------------------------------------------------------------------------
GateTLFluenceActor::GateTLFluenceActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  pMessenger = new GateTLFluenceActorMessenger(this);

  isLastHitEventImageEnabled 		= false;
  isFluenceImageEnabled 		= false;
  isEnergyFluenceImageEnabled 		= false;
  isFluenceSquaredImageEnabled		= false;
  isEnergyFluenceSquaredImageEnabled	= false;
  isFluenceUncertainyImageEnabled	= false;
  isEnergyFluenceUncertaintyImageEnabled= false;
      
  currentEvent=-1;
  nValuesPerVoxel = 1.0; // Should be possible to set by the user
  
  // This actor uses randomized hits along a track 
  SetStepHitType("random");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateTLFluenceActor::~GateTLFluenceActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateTLFluenceActor::Construct() {
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  if (!isFluenceImageEnabled &&
      !isEnergyFluenceImageEnabled) {
    GateError("The TLFluenceActor " << GetObjectName() << " does not have any image enabled ...\n Please select at least one ('enableFluence true' for example)");
  }

  // Output Filename
  fluenceFilename = G4String(removeExtension(mSaveFilename))+"-Fluence."+G4String(getExtension(mSaveFilename));
  energyFluenceFilename = G4String(removeExtension(mSaveFilename))+"-Energyfluence."+G4String(getExtension(mSaveFilename));

  SetOriginTransformAndFlagToImage(fluenceImage);
  SetOriginTransformAndFlagToImage(energyFluenceImage);


  if (isFluenceImageEnabled) {
    fluenceImage.EnableSquaredImage(isFluenceSquaredImageEnabled);
    fluenceImage.EnableUncertaintyImage(isFluenceUncertainyImageEnabled);
    fluenceImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    fluenceImage.Allocate();
    fluenceImage.SetFilename(fluenceFilename);
    fluenceImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
    fluenceImage.SetOrigin(mOrigin);
  }

  if (isEnergyFluenceImageEnabled) {
    energyFluenceImage.EnableSquaredImage(isEnergyFluenceSquaredImageEnabled);
    energyFluenceImage.EnableUncertaintyImage(isEnergyFluenceUncertaintyImageEnabled);
    energyFluenceImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    energyFluenceImage.Allocate();
    energyFluenceImage.SetFilename(energyFluenceFilename);
    energyFluenceImage.SetOverWriteFilesFlag(mOverWriteFilesFlag);
    energyFluenceImage.SetOrigin(mOrigin);
  }
  
  if (isEnergyFluenceSquaredImageEnabled || isEnergyFluenceUncertaintyImageEnabled || isFluenceSquaredImageEnabled || isEnergyFluenceUncertaintyImageEnabled)
  {
    lastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    lastHitEventImage.Allocate();
    isLastHitEventImageEnabled = true;
  }


  voxelVolume = GetDoselVolume();
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateTLFluenceActor::SaveData() {
  GateVActor::SaveData();
  if (isFluenceImageEnabled) fluenceImage.SaveData(currentEvent+1, false);
  if (isEnergyFluenceImageEnabled) energyFluenceImage.SaveData(currentEvent+1, false);
  if (isLastHitEventImageEnabled) {
    lastHitEventImage.Fill(-1); // reset
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateTLFluenceActor::ResetData() {
  if (isLastHitEventImageEnabled) lastHitEventImage.Fill(-1);
  if (isFluenceImageEnabled) fluenceImage.Reset();
  if (isEnergyFluenceImageEnabled) energyFluenceImage.Reset();
}
//-----------------------------------------------------------------------------

void GateTLFluenceActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  // Find the lengt of the step
  G4double stepLength = step->GetStepLength();
  
  // Find the average energy along the step. Not 100% sure of this.
  G4double energy = step->GetPreStepPoint()->GetKineticEnergy() - step->GetTotalEnergyDeposit() / 2.0; 
  
  // Find the direction of the step. 
  // OBS the direction is in the global coordsys. Should be changed!
  G4ThreeVector direction = step->GetPreStepPoint()->GetMomentumDirection();
  G4TouchableHandle theTouchable = step->GetPreStepPoint()->GetTouchableHandle();
  direction = theTouchable->GetHistory()->GetTopTransform().TransformAxis(direction);
  
  // Find the typical length the track has when passing a voxel
  G4double totalVoxelLength = -1;
  
  if (direction.getX() != 0.0)
   totalVoxelLength = mVoxelSize.getX()/std::abs(direction.getX());
  if (direction.getY() != 0.0)
  {
   G4double tmp = mVoxelSize.getY()/std::abs(direction.getY());
   if (tmp < totalVoxelLength)
    totalVoxelLength = tmp;
  }
  if (direction.getZ() != 0.0)
  {
   G4double tmp = mVoxelSize.getZ()/std::abs(direction.getZ());
   if (tmp < totalVoxelLength)
    totalVoxelLength = tmp;
  }
  // This shouldnt happen but better check it
  if (totalVoxelLength == -1)
    GateError("Discovered a step without direction!!");
  
  // std::cout << "Voxellength = " << totalVoxelLength <<std::endl;
  
  
  // Find the total number (n) of random values to be used along the step to 
  // associate flunce with individual voxels.
  int nStepFractions = ceil(nValuesPerVoxel*stepLength/totalVoxelLength);
  
  // Find the fluence/e-fluence of one fraction of a step. 
  G4double fluenceFraction = step->GetPreStepPoint()->GetWeight() * (stepLength/(double(nStepFractions)*voxelVolume))*(cm*cm); // In units of cm⁻2
  G4double energyFluenceFraction = energy*fluenceFraction/MeV; // In units of MEV*cm⁻2
  
 
  // Store nStepFractions fractional hits at random positions along the trajectory.
  for (int ii = 0; ii < nStepFractions; ii++)
    storeFluenceAtCurrentIndex(GetIndexFromStepPosition(GetVolume(), step), fluenceFraction, energyFluenceFraction);
}
//-----------------------------------------------------------------------------


void GateTLFluenceActor::storeFluenceAtCurrentIndex(int index,G4double fluence, G4double energyFluence)
{
   // Find out if we are in the same event as last call of this function. To ensure that
  // correlated depositions of are not counted as many events. That would give incorrect
  // uncertainty estimates.
  
  bool sameEvent=true;
  if (isLastHitEventImageEnabled) {
    if (currentEvent != lastHitEventImage.GetValue(index)) {
      sameEvent = false;
      lastHitEventImage.SetValue(index, currentEvent);
    }
  }
  
  
  // Store images
  if (isFluenceImageEnabled)
  {
    if (isFluenceSquaredImageEnabled || isFluenceUncertainyImageEnabled) 
    {
      if (sameEvent) fluenceImage.AddTempValue(index, fluence);
	else fluenceImage.AddValueAndUpdate(index, fluence);
    }
    else fluenceImage.AddValue(index, fluence);
  }
  
  if (isEnergyFluenceImageEnabled)
  {
    if (isEnergyFluenceSquaredImageEnabled || isEnergyFluenceUncertaintyImageEnabled) 
    {
      if (sameEvent) energyFluenceImage.AddTempValue(index, fluence);
	else energyFluenceImage.AddValueAndUpdate(index, energyFluence);
    }
    else energyFluenceImage.AddValue(index, fluence);
  }
  
}


//-----------------------------------------------------------------------------
void GateTLFluenceActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateTLFluenceActor -- Begin of Run\n");
  // ResetData(); // Do no reset here !! (when multiple run);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateTLFluenceActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  currentEvent++;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/* void GateTLFluenceActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
 
  G4StepPoint *PreStep(step->GetPreStepPoint());
  G4StepPoint *PostStep(step->GetPostStepPoint());
  G4ThreeVector prePosition=PreStep->GetPosition();
  G4ThreeVector postPosition=PostStep->GetPosition();
  if(step->GetTrack()->GetDefinition()->GetParticleName() == "gamma"){
    G4double distance = step->GetStepLength();
    std::cout <<"Distance = " << distance * mm<< std::endl;
    G4double energy = PreStep->GetKineticEnergy();
    double muenOverRho = mMaterialHandler->GetMuEnOverRho(PreStep->GetMaterialCutsCouple(), energy);
    G4double dose = ConversionFactor*energy*muenOverRho*distance/VoxelVolume;
    G4double edep = 0.1*energy*muenOverRho*distance*PreStep->GetMaterial()->GetDensity()/(g/cm3);
    bool sameEvent=true;

    if (mIsLastHitEventImageEnabled) {
      if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
	sameEvent = false;
	mLastHitEventImage.SetValue(index, mCurrentEvent);
      }
    }

    if(energy <= .001){
      edep=energy;
      step->GetTrack()->SetTrackStatus(fStopAndKill);
    }

    if (mIsDoseImageEnabled) {
      if (mIsDoseUncertaintyImageEnabled || mIsDoseSquaredImageEnabled) {
        if (sameEvent) mDoseImage.AddTempValue(index, dose);
        else mDoseImage.AddValueAndUpdate(index, dose);
      }
      else
        mDoseImage.AddValue(index, dose);
    }
    
    
    if(mIsEdepImageEnabled){
      if (mIsEdepUncertaintyImageEnabled || mIsEdepSquaredImageEnabled) {
	if (sameEvent) mEdepImage.AddTempValue(index, edep);
	else mEdepImage.AddValueAndUpdate(index, edep);
      }
      else
	mEdepImage.AddValue(index, edep);
    }

  }
  
}*/
//-----------------------------------------------------------------------------

#endif /* end #define GATETLEDOSEACTOR_CC */
