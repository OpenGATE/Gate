/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateDetectorInOutActorMessenger.hh"
#include "GateDetectorInOutActor.hh"

//-----------------------------------------------------------------------------
GateDetectorInOutActorMessenger::GateDetectorInOutActorMessenger(GateDetectorInOutActor* sensor):
  GateActorMessenger(sensor), pDIOActor(sensor)
{
  DDF();
  BuildCommands(baseName + sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDetectorInOutActorMessenger::~GateDetectorInOutActorMessenger()
{
  delete pSetInputPlaneCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActorMessenger::BuildCommands(G4String base)
{
  G4String n = base + "/setInputPlane";
  pSetInputPlaneCmd = new G4UIcmdWithAString(n, this);
  G4String guid = G4String("Set the volume name of the input plane");
  pSetInputPlaneCmd->SetGuidance(guid);

  n = base + "/setOutputSystem";
  pSetOutputSystemNameCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Set the name of the output system");
  pSetOutputSystemNameCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetInputPlaneCmd) pDIOActor->SetInputPlaneName(newValue);
  if (cmd == pSetOutputSystemNameCmd) pDIOActor->SetOutputSystemName(newValue);

  GateActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------
