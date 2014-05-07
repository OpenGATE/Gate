
#include "GatePhysicsExtractorActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GatePhysicsExtractorActor.hh"

GatePhysicsExtractorActorMessenger::GatePhysicsExtractorActorMessenger(GatePhysicsExtractorActor* v)
: GateActorMessenger(v), pActor(v)
{
	BuildCommands(baseName+pActor->GetObjectName());
}

GatePhysicsExtractorActorMessenger::~GatePhysicsExtractorActorMessenger()
{
	delete pProcessNameCmd;
	delete pParticleNameCmd;
	delete pMaterialNameCmd;
	delete pNbPtsCmd;
	delete pEminCmd;
	delete pEmaxCmd;
	delete pEnergyCutCmd;
}

void GatePhysicsExtractorActorMessenger::BuildCommands(G4String base)
{
	{
		pProcessNameCmd = new G4UIcmdWithAString((base+"/setProcess").c_str(),this);
		pProcessNameCmd->SetGuidance("Set process of interest");
		pProcessNameCmd->SetParameterName("Process",false);
	} {
		pParticleNameCmd = new G4UIcmdWithAString((base+"/setParticle").c_str(),this);
		pParticleNameCmd->SetGuidance("Set particle of interest");
		pParticleNameCmd->SetParameterName("Particle",false);
	} {
		pMaterialNameCmd = new G4UIcmdWithAString((base+"/setMaterial").c_str(),this);
		pMaterialNameCmd->SetGuidance("Set material of interest");
		pMaterialNameCmd->SetParameterName("Material",false);
	} {
		pNbPtsCmd = new G4UIcmdWithADouble((base+"/setNbPts").c_str(),this);
		pNbPtsCmd->SetGuidance("Set number of points per order of magnitude");
		pNbPtsCmd->SetParameterName("NbPts",false);
	} {
		pEminCmd = new G4UIcmdWithADoubleAndUnit((base+"/setEmin").c_str(),this);
		pEminCmd->SetGuidance("Set minimum energy");
		pEminCmd->SetParameterName("Emin",false);
		pEminCmd->SetUnitCategory("Energy");
	} {
		pEmaxCmd = new G4UIcmdWithADoubleAndUnit((base+"/setEmax").c_str(),this);
		pEmaxCmd->SetGuidance("Set maximum energy");
		pEmaxCmd->SetParameterName("Emax",false);
		pEmaxCmd->SetUnitCategory("Energy");
	} {
		pEnergyCutCmd = new G4UIcmdWithADoubleAndUnit((base+"/setEnergyCut").c_str(),this);
		pEnergyCutCmd->SetGuidance("Set maximum energy");
		pEnergyCutCmd->SetParameterName("ECut",false);
		pEnergyCutCmd->SetUnitCategory("Energy");
	}
}

void GatePhysicsExtractorActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
	if (cmd == pProcessNameCmd) pActor->SetProcessName(newValue);
	if (cmd == pParticleNameCmd) pActor->SetParticleName(newValue);
	if (cmd == pMaterialNameCmd) pActor->SetMaterialName(newValue);
	if (cmd == pNbPtsCmd) pActor->SetNumberOfPointsPerOrderOfMagnitude(G4UIcmdWithADouble::GetNewDoubleValue(newValue));
	if (cmd == pEminCmd) pActor->SetMinEnergy(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
	if (cmd == pEmaxCmd) pActor->SetMaxEnergy(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
	if (cmd == pEnergyCutCmd) pActor->SetEnergyCutThreshold(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
	GateActorMessenger::SetNewValue(cmd,newValue);
}

#endif
