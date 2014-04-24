/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaProductionTLEActor.hh"
#include "GatePromptGammaProductionTLEActorMessenger.hh"
#include "GateImageOfHistograms.hh"

//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActor::GatePromptGammaProductionTLEActor(G4String name,
                                                                     G4int depth):
  GateVImageActor(name,depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaProductionTLEActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActor::~GatePromptGammaProductionTLEActor()
{
  DD("GatePromptGammaProductionTLEActor:: destructor");
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::SetInputDataFilename(std::string filename)
{
  mInputDataFilename = filename;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::Construct()
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


  // Set to zero
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::ResetData()
{
  DD("GPGPTLE::ResetData");
  mImageGamma.Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::SaveData()
{
  DD("GPGPTLE::SaveData");
  GateVImageActor::SaveData();
  DD(mOverWriteFilesFlag);
  DD(mSaveFilename);
  mImageGamma.Write(mSaveFilename);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::UserPostTrackActionInVoxel(const int index, const G4Track* t)
{
  //  GateVImageActor::UserPostTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::UserPreTrackActionInVoxel(const int index, const G4Track* t)
{
  //GateVImageActor::UserPreTrackActionInVoxel(index, t);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActor::UserSteppingActionInVoxel(int index, const G4Step * step)
{
  // Check index
  if (index <0) return;

  // Get information
  const G4String & particle_name = step->GetTrack()->GetParticleDefinition()->GetParticleName();
  const G4double & particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double & distance = step->GetStepLength();

  // Check particle type ("proton")
  if (particle_name != "proton") return;

  // Check material
  // FIXME
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  G4String materialName = material->GetName();
  if (materialName == "Air_0") return;

  DD("---------------------");
  DD(index);
  DD(materialName);
  DD(particle_energy/MeV);
  DD(distance);

  // Get value from histogram
  TH1D * h = data.GetGammaEnergySpectrum(particle_energy);
  //DD(h->GetEntries());
  //  h->Print("range");
  h->Scale(distance);
  mImageGamma.AddValue(index, h);
}
//-----------------------------------------------------------------------------
