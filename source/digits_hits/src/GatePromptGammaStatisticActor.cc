/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details

  2015-03-06 Brent Huisman: Added GammaZ and Ngamma TH2Ds, to provide output that corresponds 1:1 to Jean Michels formalism.
                            In addition, the final step in the computation of GammaZ (dividing EpEpgNorm by EpInelatsic) was moved from GatePromptGammaData to here.
                            In addition, the output is no longer divided by the number of primaries (the number used to generate the material databases), as that is not necessary.
  ----------------------*/

#include "GatePromptGammaStatisticActor.hh"
#include "GatePromptGammaStatisticActorMessenger.hh"

#include <G4VProcess.hh>
#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>
#include <G4HadronicProcessStore.hh>

//-----------------------------------------------------------------------------
GatePromptGammaStatisticActor::
GatePromptGammaStatisticActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  pMessenger = new GatePromptGammaStatisticActorMessenger(this);
  sigma_filled = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaStatisticActor::~GatePromptGammaStatisticActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActor::SetProtonEMin(G4double x) { data.SetProtonEMin(x); }
void GatePromptGammaStatisticActor::SetProtonEMax(G4double x) { data.SetProtonEMax(x); }
void GatePromptGammaStatisticActor::SetGammaEMin(G4double x)  { data.SetGammaEMin(x); }
void GatePromptGammaStatisticActor::SetGammaEMax(G4double x)  { data.SetGammaEMax(x); }
void GatePromptGammaStatisticActor::SetProtonNbBins(G4int x)  { data.SetProtonNbBins(x); }
void GatePromptGammaStatisticActor::SetGammaNbBins(G4int x)   { data.SetGammaNbBins(x); }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // Create histograms
  const G4Material * m = mVolume->GetMaterial();
  data.Initialize(mSaveFilename, m);

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActor::SaveData()
{
  // Normalisation of 2D histo (pHEpEpgNormalized = E proton vs E gamma) according to
  // proba of inelastic interaction by E proton (pHEpInelastic);
  double temp=0.0;
  double temp2=0.0;
  for( int i=1; i <= data.GetGammaZ()->GetNbinsX(); i++) {
    for( int j = 1; j <= data.GetGammaZ()->GetNbinsY(); j++) {
      if(data.GetHEpInelastic()->GetBinContent(i) != 0) {
        temp = data.GetGammaZ()->GetBinContent(i,j)/data.GetHEpInelastic()->GetBinContent(i);
        temp2 = data.GetHEpEpgNormalized()->GetBinContent(i,j)/data.GetHEpInelastic()->GetBinContent(i); //for backwards compatibility
      }
      else {
        if (data.GetGammaZ()->GetBinContent(i,j)> 0.) {
          DD(i);
          DD(j);
          DD(data.GetHEpInelastic()->GetBinContent(i));
          DD(data.GetGammaZ()->GetBinContent(i,j));
          GateError("ERROR in in histograms, pHEpInelastic is zero and not pHEpEpgNormalized,GammaZ");
        }
        if (data.GetHEpEpgNormalized()->GetBinContent(i,j)> 0.) {
          DD(i);
          DD(j);
          DD(data.GetHEpInelastic()->GetBinContent(i));
          DD(data.GetHEpEpgNormalized()->GetBinContent(i,j));
          GateError("ERROR in in histograms, pHEpInelastic is zero and not pHEpEpgNormalized,GammaZ");
        }
      }
      data.GetGammaZ()->SetBinContent(i,j,temp);
      data.GetHEpEpgNormalized()->SetBinContent(i,j,temp2);
    }
  }
  data.SaveData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActor::ResetData()
{
  data.ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActor::UserSteppingAction(const GateVVolume*,
                                                       const G4Step* step)
{
  // Get various information on the current step
  const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
  const G4double particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  static G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  static G4VProcess * protonInelastic = store->FindProcess(G4Proton::Proton(), fHadronInelastic);

  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;

  // Incident Proton Energy spectrum
  data.GetHEp()->Fill(particle_energy/MeV);

  // Process type, store cross_section for ProtonInelastic process
  if (process != protonInelastic) return;
  G4double cross_section = store->GetCrossSectionPerVolume(particle, particle_energy, process, material);//(\kappa_{inel})

  data.GetHEpInelastic()->Fill(particle_energy/MeV);//N_{inel}

  // Only once : cross section of ProtonInelastic in that material
  if (!sigma_filled) {
    for (int bin = 1; bin < data.GetHEpSigmaInelastic()->GetNbinsX()+1; bin++) {
      G4double local_energy = data.GetHEpSigmaInelastic()->GetBinCenter(bin)*MeV;  //bincenter is convert to

      //DD(local_energy); // Check this

      const G4double cross_section_local = store->GetCrossSectionPerVolume(particle, local_energy, process, material);
      data.GetHEpSigmaInelastic()->SetBinContent(bin,cross_section_local);
    }
    sigma_filled = true;
  }

  // For all secondaries, store Energy spectrum
  G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();
  unsigned int produced_gamma = 0;
  for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++) {
    if ((*fSecondary)[lp1]->GetDefinition() == G4Gamma::Gamma()) {
      const double e = (*fSecondary)[lp1]->GetKineticEnergy()/MeV;
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
      data.GetHEpEpg()->Fill(particle_energy/MeV, e);
      data.GetNgamma()->Fill(particle_energy/MeV, e);//N_{\gamma}
      data.GetHEpEpgNormalized()->Fill(particle_energy/MeV, e, cross_section);
      data.GetGammaZ()->Fill(particle_energy/MeV, e, cross_section);  //so we score cross_section*1 (\kappa_{inel}*N_{\gamma}) as func of Ep,Epg
        //it stores Ngamma(which is 1) multiplied y crosssection. Divide at the end by EpInelastic to obtain GammaZ/rho(Z)
      produced_gamma++;
    }
  }
  if (produced_gamma != 0) data.GetHEpInelasticProducedGamma()->Fill(particle_energy/MeV);
}
//-----------------------------------------------------------------------------
