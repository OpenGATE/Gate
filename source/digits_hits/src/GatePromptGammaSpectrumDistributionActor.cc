/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePromptGammaSpectrumDistributionActor.hh"
#include "GatePromptGammaSpectrumDistributionActorMessenger.hh"

#include <G4VProcess.hh>
#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>
#include <G4HadronicProcessStore.hh>

//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActor::
GatePromptGammaSpectrumDistributionActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  pMessenger = new GatePromptGammaSpectrumDistributionActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActor::~GatePromptGammaSpectrumDistributionActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::SetProtonEMin(G4double x) { data.SetProtonEMin(x); }
void GatePromptGammaSpectrumDistributionActor::SetProtonEMax(G4double x) { data.SetProtonEMax(x); }
void GatePromptGammaSpectrumDistributionActor::SetGammaEMin(G4double x)  { data.SetGammaEMin(x); }
void GatePromptGammaSpectrumDistributionActor::SetGammaEMax(G4double x)  { data.SetGammaEMax(x); }
void GatePromptGammaSpectrumDistributionActor::SetProtonNbBins(G4int x)  { data.SetProtonNbBins(x); }
void GatePromptGammaSpectrumDistributionActor::SetGammaNbBins(G4int x)   { data.SetGammaNbBins(x); }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::Construct()
{
  DD("GPGSDA::Construct");
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Create histograms
  data.Initialize(mSaveFilename);

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::SaveData()
{
  DD("GPGSDA::SaveData");
  data.SaveData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::ResetData()
{
  DD("GPGSDA::ResetData");
  data.ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::UserSteppingAction(const GateVVolume*,
                                                                  const G4Step* step)
{
  // Get various information
  const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
  //const G4String particle_name = particle->GetParticleName();
  const G4double particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  static G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  static G4VProcess * protonInelastic = store->FindProcess(G4Proton::Proton(), fHadronInelastic);
  // const G4String process_name = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();

  // Check particle type ("proton")
  //if (particle_name != "proton") return;
  if (particle != G4Proton::Proton()) return;

  // Incident Proton Energy spectrum
  data.GetHEp()->Fill(particle_energy/MeV);

  // Process type, store cross_section for ProtonInelastic process
  //  if (process_name != "ProtonInelastic") return;
  if (process != protonInelastic) return;
  G4double cross_section = store->GetCrossSectionPerVolume(particle, particle_energy, process, material);
  data.GetHEpInelastic()->Fill(particle_energy/MeV);

  // Only once : cross section of ProtonInelastic in that material
  static bool sigma_filled=false;
  if (!sigma_filled) {
    for (int bin = 1; bin < data.GetHEpSigmaInelastic()->GetNbinsX()+1; bin++) {
      G4double local_energy = data.GetHEpSigmaInelastic()->GetBinCenter(bin)*MeV;
      const G4double cross_section_local = store->GetCrossSectionPerVolume(particle, local_energy, process, material);
      data.GetHEpSigmaInelastic()->SetBinContent(bin,cross_section_local);
    }
    sigma_filled = true;
  }

  // For all secondaries, store Energy spectrum
  G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();
  unsigned int produced_gamma = 0;
  for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++) {
    if ((*fSecondary)[lp1]->GetDefinition() == G4Gamma::Gamma()) { // ->GetParticleName() == "gamma") {
      const double e = (*fSecondary)[lp1]->GetKineticEnergy()/MeV;
      data.GetHEpEpg()->Fill(particle_energy/MeV, e);
      data.GetHEpEpgNormalized()->Fill(particle_energy/MeV, e, cross_section);
      produced_gamma++;
    }
  }

  if (produced_gamma != 0) data.GetHEpInelasticProducedGamma()->Fill(particle_energy/MeV);

}
//-----------------------------------------------------------------------------
