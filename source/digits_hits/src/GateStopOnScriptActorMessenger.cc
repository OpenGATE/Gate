/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATESTOPONSCRIPTACTORMESSENGER_CC
#define GATESTOPONSCRIPTACTORMESSENGER_CC

#include "GateStopOnScriptActorMessenger.hh"
#include "G4UIcmdWithABool.hh"
#include "GateStopOnScriptActor.hh"

//-----------------------------------------------------------------------------
GateStopOnScriptActorMessenger::GateStopOnScriptActorMessenger(GateStopOnScriptActor* sensor)
  :GateActorMessenger(sensor), pActor(sensor)
{
  baseName = "/gate/actor/";
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateStopOnScriptActorMessenger::~GateStopOnScriptActorMessenger()
{
  delete pEnableSaveAllActorsCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateStopOnScriptActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/saveAllActors";
  pEnableSaveAllActorsCmd = new G4UIcmdWithABool(bb,this);
  G4String guidance = "Save all actors each time this actor is saved";
  pSetFileNameCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateStopOnScriptActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if (command == pEnableSaveAllActorsCmd)
    pActor->EnableSaveAllActors(pEnableSaveAllActorsCmd->GetNewBoolValue(param));
  GateActorMessenger::SetNewValue(command, param);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEACTORMESSENGER_CC */
