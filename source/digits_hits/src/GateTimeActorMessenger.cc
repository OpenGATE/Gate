/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATETIMEACTORMESSENGER_CC
#define GATETIMEACTORMESSENGER_CC

#include "GateTimeActorMessenger.hh"
#include "GateTimeActor.hh"

//-----------------------------------------------------------------------------
GateTimeActorMessenger::GateTimeActorMessenger(GateTimeActor* sensor)
  :GateActorMessenger(sensor),
  pTimeActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTimeActorMessenger::~GateTimeActorMessenger()
{
  if(pEnableDetailedStatCmd) delete pEnableDetailedStatCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/enableDetailedStats";
  pEnableDetailedStatCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable detailed stats (slower)");
  pEnableDetailedStatCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTimeActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDetailedStatCmd)
    pTimeActor->EnableDetailedStats(pEnableDetailedStatCmd->GetNewBoolValue(newValue));
  GateActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATETIMEACTORMESSENGER_CC */
