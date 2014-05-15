/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \brief Class GateWashOutActorMessenger
*/

#ifndef GATEWASHOUTACTORMESSENGER_CC
#define GATEWASHOUTACTORMESSENGER_CC

#include "GateWashOutActorMessenger.hh"
#include "GateWashOutActor.hh"

#include "GateVActor.hh"
#include "GateActorMessenger.hh"

#include "GateUIcmdWithAStringADoubleAndADoubleWithUnit.hh"

//-----------------------------------------------------------------------------
GateWashOutActorMessenger::GateWashOutActorMessenger(GateWashOutActor* sensor)
  :GateActorMessenger(sensor),
  pWashOutActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateWashOutActorMessenger::~GateWashOutActorMessenger()
{
  delete ReadWashOutTableCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActorMessenger::BuildCommands(G4String base)
{
  G4String cmdName;

  cmdName = base+"/readTable";
  ReadWashOutTableCmd = new G4UIcmdWithAString(cmdName,this);
  ReadWashOutTableCmd->SetGuidance("Reads WashOut parameters table from a file (Ratio and Half Life with units; three components)");
  ReadWashOutTableCmd->SetGuidance("1. File name");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{

  if( command == ReadWashOutTableCmd ) {
    pWashOutActor->ReadWashOutTable(newValue);
  }
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEWASHOUTACTORMESSENGER_CC */
