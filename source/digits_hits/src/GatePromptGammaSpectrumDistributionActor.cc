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
  DD("GPGSDA::Constructor");
  pMessenger = new GatePromptGammaSpectrumDistributionActorMessenger(this);
  proton_bin = 250;
  gamma_bin = 250;
  min_proton_energy = 0*MeV;
  min_gamma_energy = 0*MeV;
  max_proton_energy = 250*MeV;
  max_gamma_energy = 10*MeV;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaSpectrumDistributionActor::~GatePromptGammaSpectrumDistributionActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::SetProtonEMin(G4double x) { min_proton_energy = x; }
void GatePromptGammaSpectrumDistributionActor::SetProtonEMax(G4double x) { max_proton_energy = x; }
void GatePromptGammaSpectrumDistributionActor::SetGammaEMin(G4double x) { min_gamma_energy = x; }
void GatePromptGammaSpectrumDistributionActor::SetGammaEMax(G4double x) { max_gamma_energy = x; }
void GatePromptGammaSpectrumDistributionActor::SetProtonNbBins(G4int x) { proton_bin = x; }
void GatePromptGammaSpectrumDistributionActor::SetGammaNbBins(G4int x) { gamma_bin = x; }
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
  pTfile = new TFile(mSaveFilename,"RECREATE");

  pHEpEpg = new TH2D("EpEpg","PG count",
                     proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                     gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);
  pHEpEpg->SetXTitle("E_{proton} [MeV]");
  pHEpEpg->SetYTitle("E_{gp} [MeV]");

  pHEpEpgNormalized = new TH2D("EpEpgNorm","PG normalized by mean free path in [m]",
                               proton_bin, min_proton_energy/MeV, max_proton_energy/MeV,
                               gamma_bin, min_gamma_energy/MeV, max_gamma_energy/MeV);
  pHEpEpgNormalized->SetXTitle("E_{proton} [MeV]");
  pHEpEpgNormalized->SetYTitle("E_{gp} [MeV]");

  pHEpInelastic = new TH1D("EpInelastic","proton energy for each inelastic interaction",
                           proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEpInelastic->SetXTitle("E_{proton} [MeV]");

  pHEp = new TH1D("Ep","proton energy",proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEp->SetXTitle("E_{proton} [MeV]");

  pHEpSigmaInelastic = new TH1D("SigmaInelastic","Sigma inelastic Vs Ep",
                                proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEpSigmaInelastic->SetXTitle("E_{proton} [MeV]");

  pHEpInelasticProducedGamma = new TH1D("EpInelasticProducedGamma",
                                        "proton energy for each inelastic interaction if gamma production",
                                        proton_bin, min_proton_energy/MeV, max_proton_energy/MeV);
  pHEpInelasticProducedGamma->SetXTitle("E_{proton} [MeV]");

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::SaveData()
{
  DD("GPGSDA::SaveData");
  pTfile->Write();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::ResetData()
{
  DD("GPGSDA::ResetData");
  pHEpEpg->Reset();
  pHEpEpgNormalized->Reset();
  pHEpInelastic->Reset();
  pHEp->Reset();
  pHEpInelasticProducedGamma->Reset();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaSpectrumDistributionActor::UserSteppingAction(const GateVVolume*,
                                                                  const G4Step* step)
{
  // Get various information
  const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
  const G4String particle_name = particle->GetParticleName();
  const G4double particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  const G4String process_name = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();

  // Check particle type ("proton")
  if (particle_name != "proton") return;

  // Incident Proton Energy spectrum
  pHEp->Fill(particle_energy/MeV);

  // Process type, store cross_section for ProtonInelastic process
  if (process_name != "ProtonInelastic") return;
  G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  G4double cross_section = store->GetCrossSectionPerVolume(particle, particle_energy, process, material);
  pHEpInelastic->Fill(particle_energy/MeV);

  // Only once : cross section of ProtonInelastic in that material
  static bool sigma_filled=false;
  if (!sigma_filled) {
    for (int bin = 1; bin < pHEpSigmaInelastic->GetNbinsX()+1; bin++) {
      G4double local_energy = pHEpSigmaInelastic->GetBinCenter(bin)*MeV;
      const G4double cross_section_local = store->GetCrossSectionPerVolume(particle, local_energy, process, material);
      pHEpSigmaInelastic->SetBinContent(bin,cross_section_local);
    }
    sigma_filled = true;
  }

  // For all secondaries, store Energy spectrum
  G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();
  unsigned int produced_gamma = 0;
  for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++) {
    if ((*fSecondary)[lp1]->GetDefinition() -> GetParticleName() == "gamma") {
      pHEpEpg->Fill(particle_energy/MeV,(*fSecondary)[lp1]->GetKineticEnergy()/MeV);
      pHEpEpgNormalized->Fill(particle_energy/MeV,(*fSecondary)[lp1]->GetKineticEnergy()/MeV,cross_section);
      produced_gamma++;
    }
  }
  if (produced_gamma != 0) pHEpInelasticProducedGamma->Fill(particle_energy/MeV);
}
//-----------------------------------------------------------------------------
