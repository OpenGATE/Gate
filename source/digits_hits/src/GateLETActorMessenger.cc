/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATELETACTORMESSENGER_CC
#define GATELETACTORMESSENGER_CC

#include "GateLETActorMessenger.hh"
#include "GateLETActor.hh"

//-----------------------------------------------------------------------------
GateLETActorMessenger::GateLETActorMessenger(GateLETActor* sensor)
  :GateImageActorMessenger(sensor),
   pLETActor(sensor)
{
  pSetRestrictedCmd = 0;
  pSetDeltaRestrictedCmd= 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateLETActorMessenger::~GateLETActorMessenger()
{
  if(pSetRestrictedCmd) delete pSetRestrictedCmd;
  if(pSetDeltaRestrictedCmd) delete pSetDeltaRestrictedCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/setRestricted";
  pSetRestrictedCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable restricted LET computation (with cut)");
  pSetRestrictedCmd->SetGuidance(guid);

  n = base+"/SetDeltaRestricted";
  pSetDeltaRestrictedCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Set the delta value for restricted LET");
  pSetDeltaRestrictedCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetRestrictedCmd)
    pLETActor->SetRestrictedFlag(pSetRestrictedCmd->GetNewBoolValue(newValue));
  if (cmd == pSetDeltaRestrictedCmd)
    pLETActor->SetDeltaRestrictedValue(pSetDeltaRestrictedCmd->GetNewDoubleValue(newValue));
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATELETACTORMESSENGER_CC */
