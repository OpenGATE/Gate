/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  mSetOutputCount = false;
  alreadyHere = false;
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
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Input data
  data.Read(mInputDataFilename);
  //data.InitializeMaterial(); //we dont need the materials, only some metadata that is already extracted in Read()

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
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::SaveData()
{
  // Data are normalized by the number of primaries
  if (alreadyHere) {
    GateError("The GatePromptGammaAnalogActor has already been saved and normalized. However, it must write its results only once. Remove all 'SaveEvery' for this actor. Abort.");
  }
  // Normalisation
  if(!mSetOutputCount){
    int n = GateActorManager::GetInstance()->GetCurrentEventId() + 1; // +1 because start at zero
    double f = 1.0 / n;
    mImageGamma->Scale(f); //converts image to float
  }
  //GateVImageActor::SaveData();
  mImageGamma->Write(mSaveFilename);
  alreadyHere = true;
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
  // Nothing (but must be implemented because virtual)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActor::UserSteppingActionInVoxel(int index, const G4Step *step)
{
  // Check if we are inside the volume (YES THIS ACTUALLY NEEDS TO BE CHECKED).
  if (index<0) return;

  // Get various information on the current step
  const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  static G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  static G4VProcess * protonInelastic = store->FindProcess(G4Proton::Proton(), fHadronInelastic);
  const G4double &particle_energy = step->GetPreStepPoint()->GetKineticEnergy();

  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;

  // Process type, store cross_section for ProtonInelastic process
  if (process != protonInelastic) return;

  // Check if proton energy within bounds.
  if (particle_energy > data.GetProtonEMax()) {
    GateError("GatePromptGammaTLEActor -- Proton Energy (" << particle_energy << ") outside range of pgTLE (" << data.GetProtonEMax() << ") database! Aborting...");
  }

  // For all secondaries, check if gamma and store pg-Energy in this voxel
  G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();
  for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++) {
    if ((*fSecondary)[lp1]->GetDefinition() == G4Gamma::Gamma()) {
      const double e = (*fSecondary)[lp1]->GetKineticEnergy()/MeV;  //convert from internal unit to MeV
      if (e>data.GetGammaEMax() || e<0.040) {
        // Without this lowE filter, we encountered a high number of 1.72keV,2.597keV,7,98467keV,8.5719keV,22.139keV,25.2572keV photons. And a bunch more.
        // These possibly correspond to Si molecular fluorescence and C-C,C-N binding energies (lung?) respectively.
        // These particles are created, and destroyed in their very first step. That's why the PhaseSpaceActor does not detect them.
        // When we filter them out, this actor and PhaseSpace agree perfectly.
        // We do not understand 100% why these are created, but we do know that such lowE photons will never make it out of the body.
        // So we think it's warrented to just filter them out, because then we reach consistency with the PhaseSpace and TLE actors.
        // These particles show up with vpgTLE too (in the db, mostly in heavier elements), so we choose a limit of 40keV so that we kill exactly the lowest bin.
        continue;
      }
      //Get thet correct gammabin
      // -1 because TH1D start at 1, and end at index=size.
      int bin = data.GetGammaZ()->GetYaxis()->FindFixBin(e)-1;
      mImageGamma->AddValueInt(index, bin, 1);

      /*Some debug stuff for lowE gammas.
      GateMessage("Actor",4,"PGAn "<<"PG added."<<std::endl);
      //GateMessage("Actor",4,"PGAn "<<"EventID: "<< step->GetEvent()->GetEventID()<<std::endl);
      GateMessage("Actor",4,"PGAn "<<"Energy: " << e<<std::endl);
      GateMessage("Actor",4,"PGAn "<<"Energy Proton: " << particle_energy<<std::endl);
      GateMessage("Actor",4,"PGAn "<<"Energy [MeV]: " << e/MeV<<std::endl);
      GateMessage("Actor",4,"PGAn "<<"Energybin: " << bin<<std::endl);
      G4HadronicProcess* hproc = (G4HadronicProcess*) process;
      const G4Isotope* target = hproc->GetTargetIsotope();
      hproc->DumpPhysicsTable();
      */
    }
  }
}
//-----------------------------------------------------------------------------

