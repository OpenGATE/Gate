/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaTLEActor.hh"
#include "GatePromptGammaTLEActorMessenger.hh"
#include "GateImageOfHistograms.hh"

#include <G4Proton.hh>
#include <G4VProcess.hh>

//-----------------------------------------------------------------------------
GatePromptGammaTLEActor::GatePromptGammaTLEActor(G4String name,
                                                                     G4int depth):
  GateVImageActor(name,depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaTLEActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaTLEActor::~GatePromptGammaTLEActor()
{
  DD("GatePromptGammaTLEActor:: destructor");
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SetInputDataFilename(std::string filename)
{
  mInputDataFilename = filename;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::Construct()
{
  DD("GPGPTLE::Construct");
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Input data
  data.Read(mInputDataFilename);
  DD(data.GetHEp()->GetEntries()); // FIXME

  // Set image parameters and allocate (only mImageGamma not mImage)
  DD(mResolution);
  DD(mHalfSize);
  DD(mPosition);
  DD(mOrigin);
  DD(mImage.GetTransformMatrix());
  DD(mOverWriteFilesFlag);
  mImageGamma.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mImageGamma.SetOrigin(mOrigin);
  mImageGamma.SetTransformMatrix(mImage.GetTransformMatrix());
  DD(mImageGamma.GetNumberOfValues());
  mImageGamma.SetHistoInfo(data.GetGammaNbBins(), data.GetGammaEMin(), data.GetGammaEMax());
  mImageGamma.Allocate();
  mImageGamma.PrintInfo();

  // Force hit type
  DD(mStepHitType);
  //  SetStepHitType("pre");
  DD(mStepHitType);

  // Set to zero
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::ResetData()
{
  DD("GPGPTLE::ResetData");
  mImageGamma.Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SaveData()
{
  DD("GPGPTLE::SaveData");
  GateVImageActor::SaveData();
  DD(mOverWriteFilesFlag);
  DD(mSaveFilename);
  mImageGamma.Write(mSaveFilename);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPostTrackActionInVoxel(const int index, const G4Track* t)
{
  //  GateVImageActor::UserPostTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPreTrackActionInVoxel(const int index, const G4Track* t)
{
  //GateVImageActor::UserPreTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserSteppingActionInVoxel(int index, const G4Step * step)
{
  // Check index
  if (index <0) return;

  // Get information
  const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
  const G4double & particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double & distance = step->GetStepLength();

  // Check particle type ("proton")
  //if (particle_name != "proton") return;
  if (particle != G4Proton::Proton()) return;

  // Check material
  // FIXME
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  G4String materialName = material->GetName();
  if (materialName != "Water") return;

  // Get value from histogram
  TH1D * h = data.GetGammaEnergySpectrum(particle_energy);
  h->Scale(distance);
  mImageGamma.AddValue(index, h);

  /*
  DD("---------------------");
  DD(index);
  DD(materialName);
  DD(particle_energy/MeV);
  DD(step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName());
  DD(distance);
  DD(h->GetSumOfWeights());
  */
}
//-----------------------------------------------------------------------------
