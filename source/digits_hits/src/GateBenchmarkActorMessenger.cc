/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/
#include "GateBenchmarkActorMessenger.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateBenchmarkActor.hh"

GateBenchmarkActorMessenger::GateBenchmarkActorMessenger(GateBenchmarkActor * v)
: GateActorMessenger(v),
	pActor(v)
{

	BuildCommands(baseName+pActor->GetObjectName());

}

GateBenchmarkActorMessenger::~GateBenchmarkActorMessenger()
{
	//delete pEmaxCmd;
	//delete pEminCmd;
}

void GateBenchmarkActorMessenger::BuildCommands(G4String /*base*/)
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

void GateBenchmarkActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
	//if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
	GateActorMessenger::SetNewValue(cmd,newValue);
}

#endif
