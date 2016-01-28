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
  guid = G4String("Enable LET uncertainty calculation");
  pSetDoseToWaterCmd->SetGuidance(guid);
  
  n = base +"/setType";
  pAveragingTypeCmd = new G4UIcmdWithAString(n,this);
  guid = G4String("Sets  averaging method ('DoseAverage', 'TrackAverage'). Default is 'DoseAverage'.");
  pAveragingTypeCmd->SetGuidance(guid);

///gate/actor/addActor                  LETActor let

///gate/actor/let/attachTo    	      myphantom

///gate/actor/let/save                  output/let.mhd

///gate/actor/let/setPosition           0 0 0 cm

///gate/actor/let/setVoxelSize          2 2 2 mm

///gate/actor/let/setType               DoseAverage

///gate/actor/let/setRestricted         true

///gate/actor/let/setDeltaRestricted    1234 mm

///gate/actor/let/setDoseToWater        false

//​

//With: 

//- LET type could be "DoseAveraged" or "TrackAverage". DoseAveraged by default

//- setRestricted is true or false (false by default)

//- setDeltaRestricted is the cut value (with length units)

//- output is a 3D image (size, voxelsize and position like doseactor), or .txt or .root (like DoseActor)

//- setDoseToWater true or false (false by default)


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetRestrictedCmd)
    pLETActor->SetRestrictedFlag(pSetRestrictedCmd->GetNewBoolValue(newValue));
  if (cmd == pSetDeltaRestrictedCmd)
    pLETActor->SetDeltaRestrictedValue(pSetDeltaRestrictedCmd->GetNewDoubleValue(newValue));
    G4cout<<"this is mRestricted in messenger"<< pSetDeltaRestrictedCmd->GetNewDoubleValue(newValue)<<G4endl;
  if (cmd == pEnableLETUncertaintyCmd) pLETActor->EnableLETUncertaintyImage(pEnableLETUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pSetDoseToWaterCmd) pLETActor->SetDoseToWater(pSetDoseToWaterCmd->GetNewBoolValue(newValue));
  if (cmd == pAveragingTypeCmd) pLETActor->SetLETType(newValue);
   
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATELETACTORMESSENGER_CC */
