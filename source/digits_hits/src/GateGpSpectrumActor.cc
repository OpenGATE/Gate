#include "GateGpSpectrumActor.hh"

#include "GateGpSpectrumActorMessenger.hh"
#include <G4VProcess.hh>
#include <G4ProtonInelasticProcess.hh>
#include <G4CrossSectionDataStore.hh>

GateGpSpectrumActor::GateGpSpectrumActor(G4String name, G4int depth):
	GateVActor(name,depth)
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
	const G4int bin = 256;

	pHEpEgp = new TH2D("EpEgp","PG count",bin,0,max_proton_energy/MeV,bin,0,max_pg_energy/MeV);
	pHEpEgp->SetXTitle("E_{proton} [MeV]");
	pHEpEgp->SetYTitle("E_{gp} [MeV]");

	pHEpEgpNormalized = new TH2D("EpEgpNorm","PG normalized by mean free path in [m]",bin,0,max_proton_energy/MeV,bin,0,max_pg_energy/MeV);
	pHEpEgpNormalized->SetXTitle("E_{proton} [MeV]");
	pHEpEgpNormalized->SetYTitle("E_{gp} [MeV]");

	pHEpInelastic = new TH1D("EpInelastic","proton energy for each inelastic interaction",bin,0,max_proton_energy/MeV);
	pHEpInelastic->SetXTitle("E_{proton} [MeV]");

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
	last_secondaries_size = 0;
	first_step = true;
}

void GateGpSpectrumActor::PostUserTrackingAction(const GateVVolume*, const G4Track*)
{
}

void GateGpSpectrumActor::UserSteppingAction(const GateVVolume*, const G4Step* step)
{
	const G4String particle_name = step->GetTrack()->GetParticleDefinition()->GetParticleName();
	if (particle_name != "proton") return;

	const G4TrackVector* secondaries = step->GetSecondary();
	if (first_step)
	{
		first_step = false;
		last_secondaries_size = secondaries->size();
	}
	if (secondaries->size() == last_secondaries_size) return;
	long int created_this_step = secondaries->size()-last_secondaries_size;
	last_secondaries_size = secondaries->size();

	const G4StepPoint* point = step->GetPostStepPoint();
	assert(point);
	const G4VProcess* process = point->GetProcessDefinedStep();
	assert(process);
	const G4String process_name = process->GetProcessName();
	if (process_name != "ProtonInelastic") return;

	G4ProtonInelasticProcess* process_casted = dynamic_cast<G4ProtonInelasticProcess*>(const_cast<G4VProcess*>(process));
	G4CrossSectionDataStore* data_store = process_casted->GetCrossSectionDataStore();
	const G4double particle_energy = step->GetPreStepPoint()->GetKineticEnergy();
	const G4Material* material = step->GetPreStepPoint()->GetMaterial();
	const G4DynamicParticle* dynamic_particle = new G4DynamicParticle(step->GetTrack()->GetParticleDefinition(),step->GetPreStepPoint()->GetMomentum());
	const G4double cross_section = data_store->GetCrossSection(dynamic_particle,material); // en distance^-1

	pHEpInelastic->Fill(particle_energy/MeV);

	//G4cout
	//	<< "coucou " << particle_name << " " << particle_energy/MeV << " " << process_name << " "
	//	<< secondaries->size() << " " << created_this_step << " "
	//	<< material->GetName() << " " << dynamic_particle->GetKineticEnergy() << " "
	//	<< process_casted << " " << data_store << " " << 1/(cross_section*mm) << G4endl;

	G4bool produced_any_gamma = false;
	for (G4TrackVector::const_reverse_iterator iter=secondaries->rbegin(); iter!=secondaries->rend(); iter++)
	{
		if (!created_this_step) break;
		created_this_step--;
		if ((*iter)->GetParticleDefinition()->GetParticleName() != "gamma") continue;
		//G4cout << "    " << (*iter)->GetParticleDefinition()->GetParticleName() << " " << (*iter)->GetKineticEnergy()/MeV << G4endl;
		pHEpEgp->Fill(particle_energy/MeV,(*iter)->GetKineticEnergy()/MeV);
		pHEpEgpNormalized->Fill(particle_energy/MeV,(*iter)->GetKineticEnergy()/MeV,cross_section*meter);
		produced_any_gamma = true;
	}

	if (produced_any_gamma) pHEpInelasticProducedGamma->Fill(particle_energy/MeV);

	delete dynamic_particle;
}

#endif
