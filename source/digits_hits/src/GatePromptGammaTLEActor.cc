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
  mCurrentEvent=-1;
  mIsUncertaintyImageEnabled = false;
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
  if(mIsUncertaintyImageEnabled) EnableBeginOfEventAction(true);
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

  if(mIsUncertaintyImageEnabled) {
    G4String TLEerrFilename = G4String(removeExtension(mSaveFilename))+"-TLEerr."+G4String(getExtension(mSaveFilename));
    SetOriginTransformAndFlagToImage(TLEerr);
    //TODO Brent: Not sure if I should add a custom flag.
    TLEerr.EnableSquaredImage(true);
    TLEerr.EnableUncertaintyImage(true);
    TLEerr.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    TLEerr.Allocate();
    TLEerr.SetFilename(TLEerrFilename);
    SetOriginTransformAndFlagToImage(mLastHitEventImage);
    mLastHitEventImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mLastHitEventImage.Allocate();
  }

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
  if (mIsUncertaintyImageEnabled) {
    TLEerr.Reset();
    mLastHitEventImage.Fill(-1);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::SaveData()
{
  // Data are normalized by the number of primaries
  static bool alreadyHere = false;
  if (alreadyHere) {
    GateError("The GatePromptGammaTLEActor has already been saved and normalized. However, it must write its results only once. Remove all 'SaveEvery' for this actor. Abort.");
  }
  // Normalisation
  int n = GateActorManager::GetInstance()->GetCurrentEventId()+1; // +1 because start at zero
  double f = 1.0/n;
  mImageGamma->Scale(f);
  GateVImageActor::SaveData();
  mImageGamma->Write(mSaveFilename);
  alreadyHere = true;

  if(mIsUncertaintyImageEnabled){
    TLEerr.SaveData(mCurrentEvent+1, false);  //TODO Check, flag determines normalization

    SetOriginTransformAndFlagToImage(mLastHitEventImage);
    mLastHitEventImage.Fill(-1);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GatePromptGammaTLEActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GatePromptGammaTLEActor -- Begin of Event: "<<mCurrentEvent << G4endl);
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

  // Check if proton energy within bounds.
  double dbmax = data.GetProtonEMax();
  if(particle_energy>dbmax){
      GateError("GatePromptGammaTLEActor -- Proton Energy ("<<particle_energy<<") outside range of pgTLE ("<<dbmax<<") database! Aborting...");
  }

  // Do not scale h directly because it will be reused
  mImageGamma->AddValueDouble(index, h, distance*material->GetDensity()/(g/cm3));

  // Error calculation
  if(mIsUncertaintyImageEnabled){
    bool sameEvent=true;
    GateDebugMessage("Actor", 2,  "GatePromptGammaTLEActor -- UserSteppingActionInVoxel: Last event in index = " << mLastHitEventImage.GetValue(index) << G4endl);
    if (mCurrentEvent != mLastHitEventImage.GetValue(index)) {
      sameEvent = false;
      mLastHitEventImage.SetValue(index, mCurrentEvent);
    }
    //TODO should this store distance or distance multiplied with factors, see JM formula 10?
    if (sameEvent) TLEerr.AddTempValue(index, distance);
    else TLEerr.AddValueAndUpdate(index, distance);
  }
}
//-----------------------------------------------------------------------------
