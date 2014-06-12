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
GatePromptGammaTLEActor::GatePromptGammaTLEActor(G4String name, G4int depth):
  GateVImageActor(name,depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaTLEActorMessenger(this);
  SetStepHitType("random");
  mImageGamma = new GateImageOfHistograms("double");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaTLEActor::~GatePromptGammaTLEActor()
{
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
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Input data
  data.Read(mInputDataFilename);
  data.InitializeMaterial();

  // Set image parameters and allocate (only mImageGamma not mImage)
  mImageGamma->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mImageGamma->SetOrigin(mOrigin);
  mImageGamma->SetTransformMatrix(mImage.GetTransformMatrix());
  mImageGamma->SetHistoInfo(data.GetGammaNbBins(), data.GetGammaEMin(), data.GetGammaEMax());
  mImageGamma->Allocate();
  mImageGamma->PrintInfo();

  // Force hit type to random
  if (mStepHitType != RandomStepHitType) {
    GateWarning("Actor '" << GetName() << "' : stepHitType forced to 'random'" << std::endl);
  }
  SetStepHitType("random");

  // Set to zero
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::ResetData()
{
  mImageGamma->Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SaveData()
{
  GateVImageActor::SaveData();
  mImageGamma->Write(mSaveFilename);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPostTrackActionInVoxel(const int, const G4Track*)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPreTrackActionInVoxel(const int, const G4Track*)
{
  // Nothing (but must be implemented because virtual)
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
  if (particle != G4Proton::Proton()) return;

  // Check material
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();

  // Get value from histogram. We do not check the material index, and
  // assume everything exist (has been computed by InitializeMaterial)
  TH1D * h = data.GetGammaEnergySpectrum(material->GetIndex(), particle_energy);

  // Do not scale h directly because it will be reused
  mImageGamma->AddValueDouble(index, h, distance);
}
//-----------------------------------------------------------------------------
