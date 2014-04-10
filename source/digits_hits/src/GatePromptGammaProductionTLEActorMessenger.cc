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
:GateActorMessenger(v), pTLEActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaProductionTLEActorMessenger::~GatePromptGammaProductionTLEActorMessenger()
{

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActorMessenger::BuildCommands(G4String /*base*/)
{

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaProductionTLEActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
