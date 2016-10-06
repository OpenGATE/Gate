/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateDoseSourceActorMessenger.hh"
#include "GateDoseSourceActor.hh"

//-----------------------------------------------------------------------------
GateDoseSourceActorMessenger::
GateDoseSourceActorMessenger(GateDoseSourceActor* v)
:GateImageActorMessenger(v), pDoseSourceActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDoseSourceActorMessenger::~GateDoseSourceActorMessenger()
{
  //DD("GateDoseSourceActorMessenger destructor");
  delete bSpotIDFromSourceCmd;
  delete bLayerIDFromSourceCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/enableSpotIDFromSource";
  bSpotIDFromSourceCmd = new G4UIcmdWithAString(bb,this);
  G4String guidance = "Store the spotID of the primary particles from given source.";
  bSpotIDFromSourceCmd->SetGuidance(guidance);
  bb = base+"/enableLayerIDFromSource";
  bLayerIDFromSourceCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Store the layerID of the primary particles from given source.";
  bLayerIDFromSourceCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSourceActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == bSpotIDFromSourceCmd) {pDoseSourceActor->SetSpotIDFromSource(newValue);};
  if (cmd == bLayerIDFromSourceCmd) {pDoseSourceActor->SetLayerIDFromSource(newValue);};
  GateImageActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
