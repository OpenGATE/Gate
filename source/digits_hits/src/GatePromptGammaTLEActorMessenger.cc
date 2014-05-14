/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaTLEActor.hh"
#include "GatePromptGammaTLEActorMessenger.hh"

//-----------------------------------------------------------------------------
GatePromptGammaTLEActorMessenger::
GatePromptGammaTLEActorMessenger(GatePromptGammaTLEActor* v)
:GateImageActorMessenger(v), pTLEActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaTLEActorMessenger::~GatePromptGammaTLEActorMessenger()
{
  DD("GatePromptGammaTLEActorMessenger destructor");
  delete pSetInputDataFileCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/setInputDataFile";
  pSetInputDataFileCmd = new G4UIcmdWithAString(bb, this);
  G4String guidance = G4String("Set input root filename with proton/gamma energy 2D spectrum (obtained from PromptGammaSpectrumDistributionActor)");
  pSetInputDataFileCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetInputDataFileCmd) pTLEActor->SetInputDataFilename(newValue);
  GateImageActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
