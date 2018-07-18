/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#include "GateSingleFixedForcedDetectionActorMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

//-----------------------------------------------------------------------------
GateSingleFixedForcedDetectionActorMessenger::GateSingleFixedForcedDetectionActorMessenger(GateSingleFixedForcedDetectionActor* sensor):
  GateFixedForcedDetectionActorMessenger(sensor),pActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSingleFixedForcedDetectionActorMessenger::~GateSingleFixedForcedDetectionActorMessenger()
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSingleFixedForcedDetectionActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
 
  bb = base+"/singleInteractionFilename";
  pSetSingleInteractionFilenameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the file name for writing the result of a single interaction.";
  pSetSingleInteractionFilenameCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionType";
  pSetSingleInteractionTypeCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Set the type of the single interaction (Compton or Rayleigh).";
  pSetSingleInteractionTypeCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionPosition";
  pSetSingleInteractionPositionCmd = new G4UIcmdWith3VectorAndUnit(bb, this);
  guidance = "Set the position of the single interaction (3D).";
  pSetSingleInteractionPositionCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionDirection";
  pSetSingleInteractionDirectionCmd = new G4UIcmdWith3Vector(bb, this);
  guidance = "Set the direction of the single interaction (3D).";
  pSetSingleInteractionDirectionCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionEnergy";
  pSetSingleInteractionEnergyCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = "Set the energy of the single interaction.";
  pSetSingleInteractionEnergyCmd->SetGuidance(guidance);

  bb = base+"/singleInteractionZ";
  pSetSingleInteractionZCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = "Set the Z of the single interaction.";
  pSetSingleInteractionZCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSingleFixedForcedDetectionActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetSingleInteractionFilenameCmd) pActor->SetSingleInteractionFilename(param);
  if(command == pSetSingleInteractionTypeCmd) pActor->SetSingleInteractionType(param);
  if(command == pSetSingleInteractionPositionCmd) pActor->SetSingleInteractionPosition(pSetSingleInteractionPositionCmd->GetNew3VectorValue(param));
  if(command == pSetSingleInteractionDirectionCmd) pActor->SetSingleInteractionDirection(pSetSingleInteractionDirectionCmd->GetNew3VectorValue(param));
  if(command == pSetSingleInteractionEnergyCmd) pActor->SetSingleInteractionEnergy(pSetSingleInteractionEnergyCmd->GetNewDoubleValue(param));
  if(command == pSetSingleInteractionZCmd) pActor->SetSingleInteractionZ(pSetSingleInteractionZCmd->GetNewIntValue(param));

  GateFixedForcedDetectionActorMessenger::SetNewValue(command ,param );
}
//-----------------------------------------------------------------------------

#endif // GATE_USE_RTK
