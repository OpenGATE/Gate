/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifndef GATEPROTONNUCLEARINFORMATIONACTORMESSENGER_CC
#define GATEPROTONNUCLEARINFORMATIONACTORMESSENGER_CC

#include "GateProtonNuclearInformationActorMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

//-----------------------------------------------------------------------------
GateProtonNuclearInformationActorMessenger::GateProtonNuclearInformationActorMessenger(GateProtonNuclearInformationActor* sensor):
  GateActorMessenger(sensor),pProtonNuclearInformationActor(sensor){}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateProtonNuclearInformationActorMessenger::~GateProtonNuclearInformationActorMessenger(){}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//NOTE: We keep the Messenger structures just in case we want to add new options to the actor
void GateProtonNuclearInformationActorMessenger::BuildCommands(G4String){}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateProtonNuclearInformationActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  GateActorMessenger::SetNewValue(command, param);
}
//-----------------------------------------------------------------------------

#endif /* end #define GateProtonNuclearInformationActorMESSENGER_CC */
