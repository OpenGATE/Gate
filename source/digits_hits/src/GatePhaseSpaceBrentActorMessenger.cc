/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePhaseSpaceBrentACTORMESSENGER_CC
#define GatePhaseSpaceBrentACTORMESSENGER_CC

#include "GatePhaseSpaceBrentActorMessenger.hh"
#include "GatePhaseSpaceBrentActor.hh"

//-----------------------------------------------------------------------------
GatePhaseSpaceBrentActorMessenger::GatePhaseSpaceBrentActorMessenger(GatePhaseSpaceBrentActor* sensor)
  :GatePhaseSpaceActorMessenger(sensor),
  pBrentActor(sensor)
{

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePhaseSpaceBrentActorMessenger::~GatePhaseSpaceBrentActorMessenger()
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhaseSpaceBrentActorMessenger::BuildCommands(G4String base)
{
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhaseSpaceBrentActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{

  GateActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GatePhaseSpaceBrentACTORMESSENGER_CC */
