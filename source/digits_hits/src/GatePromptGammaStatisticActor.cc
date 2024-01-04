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
//#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>
#include <G4HadronicProcessStore.hh>
#include <G4EmCalculator.hh>

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
  double scalingfactor=0.;
  const G4Material * m = mVolume->GetMaterial();
  for( int i=1; i <= data.GetGammaZ()->GetNbinsX(); i++) {
    for( int j = 1; j <= data.GetGammaZ()->GetNbinsY(); j++) {
      if(data.GetHEpInelastic()->GetBinContent(i) != 0) {
	// Normalization by nr of inelastic and density (cf Equation 6 of ElKanawati PMB 2015)
	scalingfactor = data.GetHEpInelastic()->GetBinContent(i) * m->GetDensity() / (g/cm3);
        temp = data.GetGammaZ()->GetBinContent(i,j) / scalingfactor;
	temp2 = data.GetHEpEpgNormalized()->GetBinContent(i,j) / scalingfactor;
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
  const G4double particle_energy = step->GetPreStepPoint()->GetKineticEnergy() - step->GetTotalEnergyDeposit(); // subtract the last ionizations before proton inelastic to get the true proton energy
  const G4Material* material = step->GetPreStepPoint()->GetMaterial();
  const G4VProcess* process = step->GetPostStepPoint()->GetProcessDefinedStep();
  static G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
  static G4VProcess * protonInelastic = store->FindProcess(G4Proton::Proton(), fHadronInelastic);

  // Check particle type ("proton")
  if (particle != G4Proton::Proton()) return;

  // G4cout << "Step = " << step->GetStepLength() / mm << " mm" << G4endl; 
  
  // Incident Proton Energy spectrum
  data.GetHEp()->Fill(particle_energy/MeV);

  // Process type, store cross_section for ProtonInelastic process
  if ((process != protonInelastic)) return;
  if ((step->GetPostStepPoint()->GetKineticEnergy() > 0.)) return; // To remove the proton inelastic that are not stopped, induce steps along the beam path and do not produce PG

  // G4cout << "Energy (MeV) = " << particle_energy/MeV << " -- TotalEnergyDeposit = " << step->GetTotalEnergyDeposit()/MeV << G4endl; // check the deposited energy before the proton inelastic process

  G4double cross_section = store->GetCrossSectionPerVolume(particle, particle_energy, process, material);//(\kappa_{inel})

  // To test the protoninelastic events that are not stoppped and induce "steps" along the beam path 
  // if ((step->GetPostStepPoint()->GetKineticEnergy() > 0.)) data.GetHEpInelastic()->Fill(particle_energy/MeV);//N_{inel}
  // else return;
  data.GetHEpInelastic()->Fill(particle_energy/MeV);//N_{ine
  
  // if (step->GetPostStepPoint()->GetKineticEnergy() > 0.) 
  //   G4cout << "Ep pre = " << step->GetPreStepPoint()->GetKineticEnergy() / MeV << " & post = " << step->GetPostStepPoint()->GetKineticEnergy() / MeV << " MeV" << G4endl; 
  
  // Only once : cross section of ProtonInelastic in that material
  if (!sigma_filled) {
    for (int bin = 1; bin < data.GetHEpSigmaInelastic()->GetNbinsX()+1; bin++) {
      G4double local_energy = data.GetHEpSigmaInelastic()->GetBinCenter(bin)*MeV;  //bincenter is convert to
      //DD(local_energy); // Check this
      const G4double cross_section_local = store->GetCrossSectionPerVolume(particle, local_energy, process, material);
      data.GetHEpSigmaInelastic()->SetBinContent(bin,cross_section_local*cm);
    }
    sigma_filled = true;
  }

  G4EmCalculator emCalculator;
  G4double dEdxFull = 0.;
  
  // For all secondaries, store Energy spectrum
  G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();
  unsigned int produced_gamma = 0;
  for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++) {
    if ((*fSecondary)[lp1]->GetDefinition() == G4Gamma::Gamma()) {
      const double e = (*fSecondary)[lp1]->GetKineticEnergy()/MeV;
      // G4cout << "PG energy = " << e/MeV << "MeV" << G4endl; 
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

      dEdxFull = emCalculator.ComputeTotalDEDX(particle_energy,particle,material) / material->GetDensity(); //for a unit density. NB: divide by (cm*cm/g) to get it in cm2/g
      // use it to scale GammaZ or EpEpgNormalized to get the version in energy integration and not distance (cf Kanawati equations 2 vs 5)
      
      produced_gamma++;
    }
  }
  if (produced_gamma != 0) data.GetHEpInelasticProducedGamma()->Fill(particle_energy/MeV);
}
//-----------------------------------------------------------------------------
