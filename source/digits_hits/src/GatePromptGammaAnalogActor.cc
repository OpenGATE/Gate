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
#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>
#include <G4HadronicProcessStore.hh>

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
  mImageGamma->Allocate();
  mImageGamma->PrintInfo();

  //sole use is to aid conversion of proton energy to bin index.
  converterHist = new TH1I("Eg", "Gamma energy", data.GetGammaNbBins(), data.GetGammaEMin() / MeV, data.GetGammaEMax() / MeV);

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
    mIsFistStep = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::UserSteppingActionInVoxel(int index, const G4Step *step)
{
  if (!mIsFistStep) return; //BRENT NOTE IS DIT NODIG?

  // Check index
  if (index < 0) return;

  // Get various information on the current step
  const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  static G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  static G4VProcess * protonInelastic = store->FindProcess(G4Proton::Proton(), fHadronInelastic);

  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;

  // Process type, store cross_section for ProtonInelastic process
  if (process != protonInelastic) return;

  // For all secondaries, store Energy spectrum
  G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();
  for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++) {
    if ((*fSecondary)[lp1]->GetDefinition() == G4Gamma::Gamma()) {
      const double e = (*fSecondary)[lp1]->GetKineticEnergy()/MeV;
      int h = gammabin(e);
      if (h>=0) mImageGamma->AddValueInt(index, h, 1);
    }
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
      //nothing yet
  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Convert Proton Energy to a bin index. Negative return value means outofbounds.
int GatePromptGammaAnalogActor::gammabin(double energy) {
  // -1 because TH1D start at 1, and end at index=size.
  //NOTE! Undocumented, but outofbounds returns -1, NOT THE INDEX OF OVER/UNDERFLOW BIN!!!
  //NOTE! We don't use FindBin, because it might change the TH1!
  int bin = converterHist->Fill(energy / MeV) - 1;
  if (bin<0 || bin>249) {
      GateMessage("Actor", 3, "Gamma energy OutOfBounds. Energy: " << energy / MeV << G4endl);
      GateMessage("Actor", 3, "Gamma energy OutOfBounds. Bin: " << bin << G4endl);
      //std::cout << "BRENT Gamma energy OutOfBounds. Energy: " << energy / MeV << std::endl;
      //std::cout << "BRENT Gamma energy OutOfBounds. Bin: " << bin+1 << std::endl;
  }
  return bin;
}
