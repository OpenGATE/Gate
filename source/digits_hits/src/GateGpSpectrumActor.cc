#include "GateGpSpectrumActor.hh"

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateGpSpectrumActorMessenger.hh"
#include <G4VProcess.hh>
#include <G4ProtonInelasticProcess.hh>
#include <G4HadronElasticProcess.hh>
#include <G4CrossSectionDataStore.hh>
#include "G4HadronicProcessStore.hh"
#include <G4UnitsTable.hh>

GateGpSpectrumActor::GateGpSpectrumActor(G4String name, G4int depth):
	GateVActor(name,depth), sigma_filled(false)
{
	pMessenger = new GateGpSpectrumActorMessenger(this);
}

GateGpSpectrumActor::~GateGpSpectrumActor()
{
	delete pMessenger;
}

void GateGpSpectrumActor::Construct()
{
	GateVActor::Construct();

	// Enable callbacks
	EnableBeginOfRunAction(true);
	EnableBeginOfEventAction(true);
	EnablePreUserTrackingAction(true);
	EnablePostUserTrackingAction(true);
	EnableUserSteppingAction(true);
	EnableEndOfEventAction(true); // for save every n

	pTfile = new TFile(mSaveFilename,"RECREATE");

	const G4double max_proton_energy = 250*MeV;
	const G4double max_pg_energy = 10*MeV;
	const G4int bin = 250;

	pHEpEgp = new TH2D("EpEgp","PG count",bin,0,max_proton_energy/MeV,bin,0,max_pg_energy/MeV);
	pHEpEgp->SetXTitle("E_{proton} [MeV]");
	pHEpEgp->SetYTitle("E_{gp} [MeV]");

	pHEpEgpNormalized = new TH2D("EpEgpNorm","PG normalized by mean free path in [m]",bin,0,max_proton_energy/MeV,bin,0,max_pg_energy/MeV);
	pHEpEgpNormalized->SetXTitle("E_{proton} [MeV]");
	pHEpEgpNormalized->SetYTitle("E_{gp} [MeV]");

	pHEpInelastic = new TH1D("EpInelastic","proton energy for each inelastic interaction",bin,0,max_proton_energy/MeV);
	pHEpInelastic->SetXTitle("E_{proton} [MeV]");

	pHEp = new TH1D("Ep","proton energy",bin,0,max_proton_energy/MeV);
	pHEp->SetXTitle("E_{proton} [MeV]");

	pHEpSigmaInelastic = new TH1D("SigmaInelastic","Sigma inelastic Vs Ep",bin,0,max_proton_energy/MeV);
	pHEpSigmaInelastic->SetXTitle("E_{proton} [MeV]");

	pHEpInelasticProducedGamma = new TH1D("EpInelasticProducedGamma","proton energy for each inelastic interaction if gamma production",bin,0,max_proton_energy/MeV);
	pHEpInelasticProducedGamma->SetXTitle("E_{proton} [MeV]");

	ResetData();
}

void GateGpSpectrumActor::SaveData()
{
	pTfile->Write();
}

void GateGpSpectrumActor::ResetData()
{
	pHEpEgp->Reset();
	pHEpEgpNormalized->Reset();
	pHEpInelastic->Reset();
	pHEp->Reset();
	pHEpInelasticProducedGamma->Reset();
}

void GateGpSpectrumActor::BeginOfRunAction(const G4Run*)
{
}

void GateGpSpectrumActor::BeginOfEventAction(const G4Event*)
{
}

void GateGpSpectrumActor::EndOfEventAction(const G4Event*)
{
}

void GateGpSpectrumActor::PreUserTrackingAction(const GateVVolume*, const G4Track*)
{
}

void GateGpSpectrumActor::PostUserTrackingAction(const GateVVolume*, const G4Track*)
{
}

struct MyHack : public G4HadronicProcess
{
	static G4CrossSectionDataStore* hack(G4HadronicProcess* process) { return static_cast<MyHack*>(process)->GetCrossSectionDataStore(); }
};

void GateGpSpectrumActor::UserSteppingAction(const GateVVolume*, const G4Step* step)
{
	const G4ParticleDefinition* particle = step->GetTrack()->GetParticleDefinition();
	const G4String particle_name = particle->GetParticleName();
	if (particle_name != "proton") return;
        const G4double particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
        pHEp->Fill(particle_energy/MeV);

	G4TrackVector* fSecondary = (const_cast<G4Step *> (step))->GetfSecondary();

	long int created_this_step = fSecondary->size();
	if (created_this_step == 0) return;

	const G4StepPoint* point = step->GetPostStepPoint();
	assert(point);
	const G4VProcess* process = point->GetProcessDefinedStep();
	assert(process);
	const G4String process_name = process->GetProcessName();

        if (process_name != "ProtonInelastic") return;

	const G4Material* material = step->GetPreStepPoint()->GetMaterial();
	G4HadronicProcessStore* store = G4HadronicProcessStore::Instance();
	G4double cross_section = store->GetCrossSectionPerVolume(particle,particle_energy,process,material);
	// G4cout << process_name << " = " << cross_section * mm << " mm-1" << G4endl;

	pHEpInelastic->Fill(particle_energy/MeV);

	if (!sigma_filled)
	{
	for (int bin = 1; bin < pHEpSigmaInelastic->GetNbinsX()+1; bin++)
	{
                G4double local_energy = pHEpSigmaInelastic->GetBinCenter(bin)*MeV;
		const G4double cross_section_local = store->GetCrossSectionPerVolume(particle,local_energy,process,material);
                pHEpSigmaInelastic->SetBinContent(bin,cross_section_local);
	}
	sigma_filled = true;
	}

	G4bool produced_any_gamma = false;
	for(size_t lp1=0;lp1<(*fSecondary).size(); lp1++)
	  {
	    if ((*fSecondary)[lp1]->GetDefinition() -> GetParticleName() == "gamma")
	      {
		pHEpEgp->Fill(particle_energy/MeV,(*fSecondary)[lp1]->GetKineticEnergy()/MeV);
		pHEpEgpNormalized->Fill(particle_energy/MeV,(*fSecondary)[lp1]->GetKineticEnergy()/MeV,cross_section);
		produced_any_gamma = true;
		// G4cout << "     E = " << (*fSecondary)[lp1] -> GetKineticEnergy() << G4endl;
	      }
	  }

	if (produced_any_gamma) pHEpInelasticProducedGamma->Fill(particle_energy/MeV);

}

#endif
