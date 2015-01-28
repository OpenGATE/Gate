/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaAnalogActor.hh"
#include "GatePromptGammaAnalogActorMessenger.hh"
#include "GateImageOfHistograms.hh"

#include <G4Proton.hh>
#include <G4VProcess.hh>

//-----------------------------------------------------------------------------
GatePromptGammaAnalogActor::GatePromptGammaAnalogActor(G4String name, G4int depth):
  GateVImageActor(name, depth)
{
  mInputDataFilename = "noFilenameGiven";
  pMessenger = new GatePromptGammaAnalogActorMessenger(this);
  //SetStepHitType("random");
  mImageGamma = new GateImageOfHistograms("int");
  mCurrentEvent = -1;
  mIsUncertaintyImageEnabled = false;
  mIsFistStep = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaAnalogActor::~GatePromptGammaAnalogActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::SetInputDataFilename(std::string filename)
{
  mInputDataFilename = filename;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::Construct()
{
  GateVImageActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  if (mIsUncertaintyImageEnabled) EnableBeginOfEventAction(true);
  if (mIsUncertaintyImageEnabled) EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Set image parameters and allocate (only mImageGamma not mImage)
  mImageGamma->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mImageGamma->SetOrigin(mOrigin);
  mImageGamma->SetTransformMatrix(mImage.GetTransformMatrix());
  mImageGamma->SetHistoInfo(data.GetGammaNbBins(), data.GetGammaEMin(), data.GetGammaEMax());
  mImageGamma->Allocate(); //FIXME: At the end.
  mImageGamma->PrintInfo();

  //sole use is to aid conversion of proton energy to bin index.
  converterHist = new TH1D("Eg", "Gamma energy", data.GetGammaNbBins(), data.GetGammaEMin() / MeV, data.GetGammaEMax() / MeV);

  // Force hit type to random
  if (mStepHitType != RandomStepHitType) {
    GateWarning("Actor '" << GetName() << "' : stepHitType forced to 'random'" << std::endl);
    SetStepHitType("random");
  }

  // Set to zero
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::ResetData()
{
  mImageGamma->Reset();
  if (mIsUncertaintyImageEnabled) {
    mLastHitEventImage.Fill(-1);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::SaveData()
{
  // Data are normalized by the number of primaries
  static bool alreadyHere = false;
  if (alreadyHere) {
    GateError("The GatePromptGammaAnalogActor has already been saved and normalized. However, it must write its results only once. Remove all 'SaveEvery' for this actor. Abort.");
  }
  GateVImageActor::SaveData();
  mImageGamma->Write(mSaveFilename);
  alreadyHere = true;

  if (mIsUncertaintyImageEnabled) {

    SetOriginTransformAndFlagToImage(mLastHitEventImage);
    mLastHitEventImage.Fill(-1);

    //Export Gamma_M database
    //data.SaveGammaM(G4String(removeExtension(mSaveFilename))+"-GammaM.root");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at start of each event
void GatePromptGammaAnalogActor::BeginOfEventAction(const G4Event *e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GatePromptGammaAnalogActor -- Begin of Event: " << mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::UserPostTrackActionInVoxel(const int, const G4Track *)
{
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::UserPreTrackActionInVoxel(const int, const G4Track *)
{
    //mIsFistStep = true;

    bEmissionPointX = t->GetVertexPosition().x();
    bEmissionPointY = t->GetVertexPosition().y();
    bEmissionPointZ = t->GetVertexPosition().z();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::UserSteppingActionInVoxel(int index, const G4Step *step)
{
  if (!mIsFistStep) return; //BRENT NOTE IS DIT NODIG?
  G4StepPoint *stepPoint;
  if (mStoreOutPart || EnableAllStep) stepPoint = step->GetPostStepPoint();//TODO CHECK BRENT!!!!
  else stepPoint = step->GetPreStepPoint();

  // BRENT NOTE: here gaan we mImageGamma vullen.

  // Check index
  if (index < 0) return;

  // Get information
  const G4ParticleDefinition *particle = step->GetTrack()->GetParticleDefinition();
  const G4double &particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4double &distance = step->GetStepLength();

  // Check particle type ("photon")
  //if (particle != G4Photon::Photon()) return;
  //TODO use filters: partfilter: photon, directparent = proton, addCreatorProcess protonInelastic

  // Get value from histogram. We do not check the material index, and
  h = gammabin(particle_energy);
  energy = stepPoint->GetKineticEnergy();

  // Do not scale h
  mImageGamma->AddValueInt(index, h);

  // Error calculation
  if (mIsUncertaintyImageEnabled) {
    //tmptrackl->AddValueDouble(index, protbin(particle_energy), distance);
  }
  mIsFistStep = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback at end of each event
void GatePromptGammaAnalogActor::EndOfEventAction(const G4Event *e) {
  GateVActor::BeginOfEventAction(e);
  GateDebugMessage("Actor", 3, "GatePromptGammaAnalogActor -- End of Event: " << mCurrentEvent << G4endl);

  if (mIsUncertaintyImageEnabled) {
    double *itmptrackl = tmptrackl->GetDataDoublePointer();
    double *itrackl = trackl->GetDataDoublePointer();
    double *itracklsq = tracklsq->GetDataDoublePointer();
    for (long i = 0; i < tmptrackl->GetDoubleSize() ; i++) {
      itrackl[i] += itmptrackl[i];
      itracklsq[i] += itmptrackl[i] * itmptrackl[i];
      itmptrackl[i] = 0.; //reset for next event
    }
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Convert Proton Energy to a bin index.
int GatePromptGammaAnalogActor::gammabin(double energy) {
  return converterHist->Fill(energy / MeV);
}
