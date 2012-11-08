#include "GateGpSpectrumActor.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateGpSpectrumActorMessenger.hh"
#include <G4VProcess.hh>

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

	pHEpEgp = new TH2D("EpEgp","PG versus proton energy",100,0,250,100,0,10);
	pHEpEgp->SetXTitle("E_{proton} [MeV]");
	pHEpEgp->SetYTitle("E_{gp} [MeV]");

	ResetData();
}

void GateGpSpectrumActor::SaveData()
{
	pTfile->Write();
}

void GateGpSpectrumActor::ResetData() 
{
	pHEpEgp->Reset();
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

	const G4double particle_energy_pre = step->GetPreStepPoint()->GetKineticEnergy();
	//const G4double particle_energy_post = step->GetPostStepPoint()->GetKineticEnergy();
	const G4double particle_energy = particle_energy_pre;

	G4cout << "coucou " << particle_name << " " << particle_energy/MeV << " " << process_name << " " << secondaries->size() << " " << created_this_step << G4endl;
	for (G4TrackVector::const_reverse_iterator iter=secondaries->rbegin(); iter!=secondaries->rend(); iter++)
	{
		if (!created_this_step) break;
		created_this_step--;
		if ((*iter)->GetParticleDefinition()->GetParticleName() != "gamma") continue;
		G4cout << "    " << (*iter)->GetParticleDefinition()->GetParticleName() << " " << (*iter)->GetKineticEnergy()/MeV << G4endl;
		pHEpEgp->Fill(particle_energy/MeV,(*iter)->GetKineticEnergy()/MeV);
	}
	
}

#endif
