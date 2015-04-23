#include "GatePhysicsExtractorActor.hh"

#include <fstream>
#include <iostream>
using Gateendl;

GatePhysicsExtractorActor::GatePhysicsExtractorActor(G4String name, G4int depth)
	: GateVActor(name,depth)
{
	pCalculator = new G4EmCalculator;
	pParticleDefinition = NULL;
	pMaterial = NULL;

	pProcessName = "";
	pMinEnergy = 1*keV;
	pMaxEnergy = 20*MeV;
	pNbPoints  = 100;
	pCutThreshold = 1*keV;

	pMessenger = new GatePhysicsExtractorActorMessenger(this);
}

GatePhysicsExtractorActor::~GatePhysicsExtractorActor()
{
	delete pCalculator;
}

void GatePhysicsExtractorActor::SetParticleName(const G4String& particleName)
{
	pParticleDefinition = pCalculator->FindParticle(particleName);
	if (!pParticleDefinition) G4Exception("GatePhysicsExtractorActor::SetParticleName","SetParticleName",FatalException,("Can't find particle "+particleName).c_str());
}

void GatePhysicsExtractorActor::SetMaterialName(const G4String& materialName)
{
	pMaterial = pCalculator->FindMaterial(materialName);
	if (!pMaterial) G4Exception("GatePhysicsExtractorActor::SetMaterialName","SetMaterialName",FatalException,("Can't find particle "+materialName).c_str());
}

void GatePhysicsExtractorActor::Construct()
{
	GateVActor::Construct();
	assert(pParticleDefinition);
	assert(pMaterial);
	assert(!pProcessName.empty());

	// Enable callbacks
	EnableBeginOfRunAction(true);
	EnableBeginOfEventAction(true);
	EnablePreUserTrackingAction(false);
	EnablePostUserTrackingAction(false);
	EnableUserSteppingAction(false);
	EnableEndOfEventAction(true);

	ResetData();
}

void GatePhysicsExtractorActor::SaveData()
{
	const int n = std::ceil((log(pMaxEnergy)-log(pMinEnergy))*pNbPoints/log(10));
	const double q = pow(pMaxEnergy/pMinEnergy,1./(n-1));
	double energy = pMinEnergy;

	std::ofstream handle(mSaveFilename);
	handle << "particle = " << pParticleDefinition->GetParticleName() << endl;
	handle << "material = " << pMaterial->GetName() << endl;
	handle << "process  = " << pProcessName << endl;
	handle << "ecut     = " << pCutThreshold/keV << "keV" << endl;
	handle << "====================================================" << endl;
	handle << "Energy [MeV]\tDEDX [MeV.mm-1]\tCS per volume[mm-1]\tMean Free Path[mm]" << endl;
	for (int kk=0; kk<n; kk++)
	{
		handle << energy/MeV << "\t";
		handle << pCalculator->ComputeDEDX(energy,pParticleDefinition,pProcessName,pMaterial,pCutThreshold)/(MeV/mm) << "\t";
		handle << pCalculator->ComputeCrossSectionPerVolume(energy,pParticleDefinition,pProcessName,pMaterial,pCutThreshold)*mm << "\t";
		handle << pCalculator->ComputeMeanFreePath(energy,pParticleDefinition,pProcessName,pMaterial,pCutThreshold)/mm << endl;
		energy *= q;
	}
	handle.close();
}

void GatePhysicsExtractorActor::ResetData()
{
}

void GatePhysicsExtractorActor::BeginOfRunAction(const G4Run*)
{
}

void GatePhysicsExtractorActor::BeginOfEventAction(const G4Event*)
{
}

void GatePhysicsExtractorActor::EndOfEventAction(const G4Event*)
{
}

void GatePhysicsExtractorActor::PreUserTrackingAction(const GateVVolume*, const G4Track*)
{
}

void GatePhysicsExtractorActor::PostUserTrackingAction(const GateVVolume*, const G4Track*)
{
}

void GatePhysicsExtractorActor::UserSteppingAction(const GateVVolume*, const G4Step*)
{
}
