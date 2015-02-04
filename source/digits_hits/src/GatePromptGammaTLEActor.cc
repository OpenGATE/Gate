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
  GateVImageActor(name, depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaTLEActorMessenger(this);
  SetStepHitType("random");
  mImageGamma = new GateImageOfHistograms("double");
  mCurrentEvent = -1;
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
  if (mIsUncertaintyImageEnabled) EnableBeginOfEventAction(true);
  if (mIsUncertaintyImageEnabled) EnableEndOfEventAction(true);
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
  mImageGamma->Allocate(); //FIXME: At the end.
  mImageGamma->PrintInfo();

  //sole use is to aid conversion of proton energy to bin index.
  converterHist = new TH1D("Ep", "proton energy", data.GetProtonNbBins(), data.GetProtonEMin() / MeV, data.GetProtonEMax() / MeV);

  if (mIsUncertaintyImageEnabled) {
    //init databases for uncertainty calculations
    tmptrackl = new GateImageOfHistograms("double");
    trackl = new GateImageOfHistograms("double");
    tracklsq = new GateImageOfHistograms("double");
    tleuncertain = new GateImageOfHistograms("double");

    std::vector<GateImageOfHistograms*> list;
    //list.push_back(tleuncertain);//not needed during simulation
    list.push_back(tmptrackl);
    list.push_back(trackl);
    list.push_back(tracklsq);
    for (int i = 0; i < list.size(); i++) {
      list[i]->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
      list[i]->SetOrigin(mOrigin);
      list[i]->SetTransformMatrix(mImage.GetTransformMatrix());
      list[i]->SetHistoInfo(data.GetProtonNbBins(), data.GetProtonEMin(), data.GetProtonEMax());
      list[i]->Allocate();
      list[i]->PrintInfo();
    }
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
      tmptrackl->Reset();
      trackl->Reset();
      tracklsq->Reset();
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
  int n = GateActorManager::GetInstance()->GetCurrentEventId() + 1; // +1 because start at zero
  double f = 1.0 / n;
  mImageGamma->Scale(f);
  GateVImageActor::SaveData();
  mImageGamma->Write(mSaveFilename);
  alreadyHere = true;

  if (mIsUncertaintyImageEnabled) {
    //TLEerr.SaveData(mCurrentEvent+1, false);  //TODO Check, flag determines normalization

    SetOriginTransformAndFlagToImage(mLastHitEventImage);
    mLastHitEventImage.Fill(-1);

    //Export Gamma_M database
    //data.SaveGammaM(G4String(removeExtension(mSaveFilename))+"-GammaM.root");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at start of each event
void GatePromptGammaTLEActor::BeginOfEventAction(const G4Event *e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GatePromptGammaTLEActor -- Begin of Event: " << mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPostTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserPreTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActor::UserSteppingActionInVoxel(int index, const G4Step *step)
{
  // Check index
  if (index < 0) return;

  // Get information
  const G4ParticleDefinition *particle = step->GetTrack()->GetParticleDefinition();
  const G4double &particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double &distance = step->GetStepLength();

  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;

  // Check material
  const G4Material *material = step->GetPreStepPoint()->GetMaterial();

  // Get value from histogram. We do not check the material index, and
  // assume everything exist (has been computed by InitializeMaterial)
  TH1D *h = data.GetGammaEnergySpectrum(material->GetIndex(), particle_energy);

  // Check if proton energy within bounds.
  double dbmax = data.GetProtonEMax();
  if (particle_energy > dbmax) {
    GateError("GatePromptGammaTLEActor -- Proton Energy (" << particle_energy << ") outside range of pgTLE (" << dbmax << ") database! Aborting...");
  }

  // Do not scale h directly because it will be reused
  mImageGamma->AddValueDouble(index, h, distance * material->GetDensity() / (g / cm3));

  // Error calculation
  if (mIsUncertaintyImageEnabled) {
    //this must be moved out of this loop when it replaces recording the pg spectrum per voxel.
    /*  DD(particle_energy/MeV);
      DD(protbin(particle_energy));
      DD(index);
      DD(distance);*/
    tmptrackl->AddValueDouble(index, protbin(particle_energy), distance);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback at end of each event
void GatePromptGammaTLEActor::EndOfEventAction(const G4Event *e) {
  GateVActor::EndOfEventAction(e);
  GateDebugMessage("Actor", 3, "GatePromptGammaTLEActor -- End of Event: " << mCurrentEvent << G4endl);

    //TODO: rewrite using pattern of Doseactor to NOT use EndOfEventaction.
  if (mIsUncertaintyImageEnabled) {
    double *itmptrackl = tmptrackl->GetDataDoublePointer();
    double *itrackl = trackl->GetDataDoublePointer();
    double *itracklsq = tracklsq->GetDataDoublePointer();
    //DD(tmptrackl->GetDoubleSize() );
    for (unsigned long i = 0; i < tmptrackl->GetDoubleSize() ; i++) {

      itrackl[i] += itmptrackl[i];
      itracklsq[i] += itmptrackl[i] * itmptrackl[i];
      itmptrackl[i] = 0.; //reset for next event
    }
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Convert Proton Energy to a bin index.
int GatePromptGammaTLEActor::protbin(double energy) {
  return converterHist->Fill(energy / MeV);
}
