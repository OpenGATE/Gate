
#include "GateGpSpectrumActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateGpSpectrumActor.hh"

GateGpSpectrumActorMessenger::GateGpSpectrumActorMessenger(GateGpSpectrumActor* v)
: GateActorMessenger(v), pActor(v)
{

	BuildCommands(baseName+pActor->GetObjectName());

}

GateGpSpectrumActorMessenger::~GateGpSpectrumActorMessenger()
{
	//delete pEmaxCmd;
	//delete pEminCmd;
}

void GateGpSpectrumActorMessenger::BuildCommands(G4String /*base*/)
{
	//bb = base+"/energySpectrum/setEmin";
	//pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this); 
	//guidance = G4String("Set minimum energy of the energy spectrum");
	//pEminCmd->SetGuidance(guidance);
	//pEminCmd->SetParameterName("Emin", false);
	//pEminCmd->SetDefaultUnit("MeV");

	//bb = base+"/energyLossHisto/setNumberOfBins";
	//pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this); 
	//guidance = G4String("Set number of bins of the energy loss histogram");
	//pEdepNBinsCmd->SetGuidance(guidance);
	//pEdepNBinsCmd->SetParameterName("Nbins", false);

}

void GateGpSpectrumActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
	//if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
	GateActorMessenger::SetNewValue(cmd,newValue);
}

#endif
