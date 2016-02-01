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
  pEnableLETUncertaintyCmd = 0;
  pSetDoseToWaterCmd = 0;
  pAveragingTypeCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateLETActorMessenger::~GateLETActorMessenger()
{
  if(pSetRestrictedCmd) delete pSetRestrictedCmd;
  if(pSetDeltaRestrictedCmd) delete pSetDeltaRestrictedCmd;
  if(pEnableLETUncertaintyCmd) delete pEnableLETUncertaintyCmd;
  if(pSetDoseToWaterCmd) delete pSetDoseToWaterCmd;
  if(pAveragingTypeCmd) delete pAveragingTypeCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/setRestricted";
  pSetRestrictedCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable restricted LET computation (with cut)");
  pSetRestrictedCmd->SetGuidance(guid);
  
  n = base+"/setDeltaRestricted";
  pSetDeltaRestrictedCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Set the delta value for restricted LET");
  pSetDeltaRestrictedCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyLET";
  pEnableLETUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable LET uncertainty calculation");
  pEnableLETUncertaintyCmd->SetGuidance(guid);
  
  n = base+"/setDoseToWater";
  pSetDoseToWaterCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose-to-water correction in LET calculation");
  pSetDoseToWaterCmd->SetGuidance(guid);
  
  n = base +"/setType";
  pAveragingTypeCmd = new G4UIcmdWithAString(n,this);
  guid = G4String("Sets  averaging method ('DoseAverage', 'TrackAverage'). Default is 'DoseAverage'.");
  pAveragingTypeCmd->SetGuidance(guid);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetRestrictedCmd)
    pLETActor->SetRestrictedFlag(pSetRestrictedCmd->GetNewBoolValue(newValue));
  if (cmd == pSetDeltaRestrictedCmd)
    pLETActor->SetDeltaRestrictedValue(pSetDeltaRestrictedCmd->GetNewDoubleValue(newValue));
    //G4cout<<"this is mRestricted in messenger"<< pSetDeltaRestrictedCmd->GetNewDoubleValue(newValue)<<G4endl;
  if (cmd == pEnableLETUncertaintyCmd) pLETActor->EnableLETUncertaintyImage(pEnableLETUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pSetDoseToWaterCmd) pLETActor->SetDoseToWater(pSetDoseToWaterCmd->GetNewBoolValue(newValue));
  if (cmd == pAveragingTypeCmd) pLETActor->SetLETType(newValue);
   
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATELETACTORMESSENGER_CC */
