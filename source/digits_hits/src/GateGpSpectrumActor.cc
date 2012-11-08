#include "GateGpSpectrumActor.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateGpSpectrumActorMessenger.hh"

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
	pHEpEgp->Clear();
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

void GateGpSpectrumActor::UserSteppingAction(const GateVVolume*, const G4Step*)
{
	G4cout << "coucou" << G4endl;
}

#endif
