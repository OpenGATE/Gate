/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATECROSSSECTIONPRODUCTIONACTORMESSENGER_CC
#define GATECROSSSECTIONPRODUCTIONACTORMESSENGER_CC

#include "GateCrossSectionProductionActorMessenger.hh"

#include "GateCrossSectionProductionActor.hh"

//-----------------------------------------------------------------------------
GateCrossSectionProductionActorMessenger::GateCrossSectionProductionActorMessenger(GateCrossSectionProductionActor* sensor)
  :GateImageActorMessenger(sensor),
  pCrossSectionProductionActor(sensor)
{
pC11FilenameCmd =0;
pO15Cmd=0;
pC11Cmd=0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateCrossSectionProductionActorMessenger::~GateCrossSectionProductionActorMessenger()
{
if(pO15Cmd) delete pO15Cmd;
if(pC11Cmd) delete pC11Cmd;
if(pC11FilenameCmd)  delete pC11FilenameCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCrossSectionProductionActorMessenger::BuildCommands(G4String base)
{
  G4String cmdName=base+"/setFilename";

  pC11FilenameCmd= new G4UIcmdWithAString(cmdName , this);

  pC11FilenameCmd->SetGuidance("Set filename");

cmdName=base+"/addC11";
pC11Cmd= new G4UIcmdWithABool(cmdName , this);
pC11Cmd->SetGuidance("Add C-11 element");

cmdName=base+"/addO15";
pO15Cmd= new G4UIcmdWithABool(cmdName , this);
pO15Cmd->SetGuidance("Add O-15 element");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCrossSectionProductionActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pC11FilenameCmd) pCrossSectionProductionActor->SetFilename(newValue);
if (cmd == pC11Cmd) pCrossSectionProductionActor->ActiveC11(pC11Cmd->GetNewBoolValue(newValue));
if (cmd == pO15Cmd) pCrossSectionProductionActor->ActiveO15(pO15Cmd->GetNewBoolValue(newValue));
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEDOSEACTORMESSENGER_CC */
