/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaProductionTLEActor.hh"
#include "GatePromptGammaProductionTLEActorMessenger.hh"

//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActorMessenger::
GatePromptGammaProductionTLEActorMessenger(GatePromptGammaProductionTLEActor* v)
:GateImageActorMessenger(v), pTLEActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActorMessenger::~GatePromptGammaProductionTLEActorMessenger()
{
  DD("GatePromptGammaProductionTLEActorMessenger destructor");
  delete pSetInputDataFileCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/setInputDataFile";
  pSetInputDataFileCmd = new G4UIcmdWithAString(bb, this);
  G4String guidance = G4String("Set input root filename with proton/gamma energy 2D spectrum (obtained from PromptGammaSpectrumDistributionActor)");
  pSetInputDataFileCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetInputDataFileCmd) pTLEActor->SetInputDataFilename(newValue);
  GateImageActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
