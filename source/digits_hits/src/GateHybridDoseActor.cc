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
  mIsEdepImageEnabled = false;
  mIsDoseUncertaintyImageEnabled = false;
  
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
  pHybridMultiplicityActor->SetMultiplicity(mPrimaryMultiplicity, mSecondaryMultiplicity, attachedVolume);

  // Get the stepping manager
  mSteppingManager = G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager();
  
  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
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
      
  // Output Filename
  mDoseFilename = G4String(removeExtension(mSaveFilename))+"-Dose."+G4String(getExtension(mSaveFilename));
  mEdepFilename = G4String(removeExtension(mSaveFilename))+"-Edep."+G4String(getExtension(mSaveFilename));
//   GateMessage("Actor", 0, "taille : " << mResolution << " " << mHalfSize << " " << mPosition);
  if (mIsEdepImageEnabled) {
    mEdepImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mEdepImage.Allocate();
    mEdepImage.SetFilename(mEdepFilename);
    mEdepImage.SetOrigin(mOrigin);
  }
  mDoseImage.EnableUncertaintyImage(mIsDoseUncertaintyImageEnabled);
  mDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  //mDoseImage.SetScaleFactor(1e12/mDoseImage.GetVoxelVolume());
  mDoseImage.Allocate();
  mDoseImage.SetFilename(mDoseFilename);
  mDoseImage.SetOrigin(mOrigin);

  mPDoseFilename = G4String(removeExtension(mSaveFilename))+"-PrimaryDose."+G4String(getExtension(mSaveFilename));

  mPrimaryDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  //mDoseImage.SetScaleFactor(1e12/mDoseImage.GetVoxelVolume());
  mPrimaryDoseImage.Allocate();
  mPrimaryDoseImage.SetFilename(mPDoseFilename);
  mPrimaryDoseImage.SetOrigin(mOrigin);

  mSDoseFilename = G4String(removeExtension(mSaveFilename))+"-SecondaryDose."+G4String(getExtension(mSaveFilename));
  
  mSecondaryDoseImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  //mDoseImage.SetScaleFactor(1e12/mDoseImage.GetVoxelVolume());
  mSecondaryDoseImage.Allocate();
  mSecondaryDoseImage.SetFilename(mSDoseFilename);
  mSecondaryDoseImage.SetOrigin(mOrigin);
  //GateMessage("Actor", 0, " Activation image dose" << G4endl);
  if(mIsDoseUncertaintyImageEnabled){
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
    mIsLastHitEventImageEnabled = true;
  }
  //GateMessage("Actor", 0, " Activé  " << G4endl);
  
  ConversionFactor = 1.60217653e-19 * 1.e6 * 1.e3 * 1.e3;
  VoxelVolume = GetDoselVolume();
  outputEnergy = 0.0;
  totalEnergy = 0.0;
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateHybridDoseActor::SaveData() {
  std::cout.precision(20);
  // GateMessage("Actor", 0, " output energy : " << outputEnergy << G4endl);
  // GateMessage("Actor", 0, " total energy : " << totalEnergy << G4endl);
  // GateMessage("Actor", 0, " sum : " << totalEnergy+outputEnergy << G4endl);
  mDoseImage.SaveData(mCurrentEvent+1, false);
  mPrimaryDoseImage.SaveData(mCurrentEvent+1, false);
  mSecondaryDoseImage.SaveData(mCurrentEvent+1, false);
//   mEdepImage.SaveData(mCurrentEvent+1, false);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::ResetData() {
  mDoseImage.Reset();
  mPrimaryDoseImage.Reset();
  mSecondaryDoseImage.Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 0, "GateDoseActor -- Begin of Run" << G4endl);
  // ResetData(); // Do no reset here !! (when multiple run);
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

  // Francois : general method (slow but generic)
//   G4ParticleDefinition *hybridino = G4Hybridino::Hybridino();
//   GateVSource* source = GateSourceMgr::GetInstance()->GetSource(GateSourceMgr::GetInstance()->GetCurrentSourceID());
//   G4Event *modifiedEvent = const_cast<G4Event *>(event);
//   int initialVertexNumber = event->GetNumberOfPrimaryVertex();
//   int vertexNumber = initialVertexNumber;
// 
//   for(int i=0; i<mPrimaryMultiplicity; i++)
//   {
//     vertexNumber += source->GeneratePrimaries(modifiedEvent);    
//   }
// 
// //   GateMessage("Actor", 0, " initVertexNumber " << initialVertexNumber << " newVertexNumber " << event->GetNumberOfPrimaryVertex() << G4endl);
// //   GateMessage("Actor", 0, " particle " << source->GetParticleDefinition()->GetParticleName() << G4endl);  
//   G4PrimaryVertex *hybridVertex = modifiedEvent->GetPrimaryVertex(initialVertexNumber);
// 
//   while(hybridVertex != 0)
//   {
//     G4PrimaryParticle *hybridParticle = hybridVertex->GetPrimary();
//     while(hybridParticle != 0)
//     {
// //       GateMessage("Actor", 0, " vertex " << hybridVertex << " particle " << hybridParticle << " momentum " << hybridParticle->GetMomentumDirection() << G4endl);
//       hybridParticle->SetParticleDefinition(hybridino);
//       hybridParticle = hybridParticle->GetNext();
// 
//     }
//     hybridVertex = hybridVertex->GetNext();
//   }
  
  
    // Francois : particular method (fast but only for single particle sources)
//   G4ParticleDefinition *hybridino = G4Hybridino::Hybridino();
//   G4ParticleDefinition *gamma = G4Gamma::Gamma();
// 
//   GateVSource* source = GateSourceMgr::GetInstance()->GetSource(GateSourceMgr::GetInstance()->GetCurrentSourceID());
//   GateSPSEneDistribution* enedist = source->GetEneDist();
//   GateSPSPosDistribution* posdist = source->GetPosDist();
//   GateSPSAngDistribution* angdist = source->GetAngDist();
// 
//   G4Event *modifiedEvent = const_cast<G4Event *>(event);
//   // WARNING - G4Event cannot be modified by default because of its 'const' status.
//   // Use of the 'const_cast' function to overcome this problem.
// 
//   G4ThreeVector position;
//   G4ThreeVector momentum;
//   G4double energy;
//   for(int i=0; i<mPrimaryMultiplicity; i++)
//   {
//     position = posdist->GenerateOne();
//     momentum = angdist->GenerateOne();
//     energy = enedist->GenerateOne(gamma);
// 
//     G4PrimaryParticle *hybridParticle = new G4PrimaryParticle(hybridino,momentum.x(),momentum.y(),momentum.z(),energy);
//     hybridParticle->SetKineticEnergy(energy*MeV);
// 
//     G4PrimaryVertex *hybridVertex = new G4PrimaryVertex();
//     hybridVertex->SetPosition(position.x(),position.y(),position.z());
//     hybridVertex->SetPrimary(hybridParticle);
// 
//     modifiedEvent->AddPrimaryVertex(hybridVertex); 
//   }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::PreUserTrackingAction(const GateVVolume *, const G4Track *)
{
//    GateMessage("Actor", 0, " TOTO "<< G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridDoseActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t){
  outputEnergy+= t->GetKineticEnergy();
}
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

//   GateMessage("Actor", 0, " before : position " << position << " momentum " << momentum << G4endl);
  position = worldToVolume.TransformPoint(position);  
  momentum.transform(mRotationMatrix);
  G4double energy = preStep->GetKineticEnergy();
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
  int voxelIndex = 0;
  double doseValue = 0.0;
  double L = 0.0;
  
  double hybridTrackWeight = pHybridMultiplicityActor->GetHybridTrackWeight();
  double delta_in  = hybridTrackWeight;
  double delta_out(0.0);
  double mu(0.0);
  double muen(0.0);

//   GateMessage("ActorDose", 0, "hybridWeight = " << hybridTrackWeight << " trackWeight = " << step->GetTrack()->GetWeight() << G4endl);
  GateVImageVolume* volume = dynamic_cast<GateVImageVolume*>(GetVolume());
  if(!volume) { GateError("Error in " << GetName() << ": GateVImageVolume doesn't exist"); }
  G4Material* material = preStep->GetMaterial();
  G4bool isPrimaryParticle = false;
  if(step->GetTrack()->GetParentID() == 0) { isPrimaryParticle = true; }
  
  while(L < Ltot-0.00001)
  {
//     GateMessage("Actor", 0, " index " << x << " " << y << " " << z << " | Rest " << Rx << " " << Ry << " " << Rz << " | L " << L << " Ltot " << Ltot << G4endl);
//     GateMessage("Actor", 0, " Rest  " << Rx << " " << Ry << " " << Rz << G4endl);
//     GateMessage("Actor", 0, "L " << L << " Ltot " << Ltot << G4endl);
//     GateMessage("Actor", 0, "Volume = " << GetVolume()->GetLogicalVolumeName() << G4endl);
//     GateMessage("Actor", 0, "material 1 : " << volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(x,y,z)) << G4endl);    
//     GateMessage("Actor", 0, "label : " << volume->GetImage()->GetValue(0,0,0) << " " << volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(0,0,0)) << G4endl);    

    material = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(volume->GetMaterialNameFromLabel(volume->GetImage()->GetValue(x,y,z)));
//     GateMessage("Actor", 0, "material name : " << material->GetName() << G4endl);
    mu = mMaterialHandler->GetMu(material, energy)*material->GetDensity()/(g/cm3);
    muen = mMaterialHandler->GetAttenuation(material, energy);
//     GateMessage("Actor", 0, " material " << material->GetName() << " energy " << energy << " mu " << mu << " muen " << muen << " voxVol " << VoxelVolume << G4endl);

    voxelIndex = x+y*lineSize+z*planeSize;
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
    
    doseValue = ConversionFactor*energy*muen*(delta_in-delta_out)/mu/VoxelVolume;
    
    if(isPrimaryParticle) {
      mDoseImage.AddValue(voxelIndex, doseValue);
      mPrimaryDoseImage.AddValue(voxelIndex, doseValue);
    }
    else {
      mDoseImage.AddValue(voxelIndex, doseValue);
      mSecondaryDoseImage.AddValue(voxelIndex, doseValue);
    }
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

