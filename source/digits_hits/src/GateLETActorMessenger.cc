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
  pEnableLETUncertaintyCmd = 0;
  pSetLETtoWaterCmd = 0;
  pAveragingTypeCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateLETActorMessenger::~GateLETActorMessenger()
{
  if(pEnableLETUncertaintyCmd) delete pEnableLETUncertaintyCmd;
  if(pSetLETtoWaterCmd) delete pSetLETtoWaterCmd;
  if(pAveragingTypeCmd) delete pAveragingTypeCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/enableUncertaintyLET";
  pEnableLETUncertaintyCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable LET uncertainty calculation");
  pEnableLETUncertaintyCmd->SetGuidance(guid);

  n = base+"/setLETtoWater";
  pSetLETtoWaterCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose-to-water correction in LET calculation");
  pSetLETtoWaterCmd->SetGuidance(guid);

  n = base+"/doParallelCalculation";
  pSetParallelCalculationCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable parallel calculation: creates 2 output files for each simulation");
  pSetParallelCalculationCmd->SetGuidance(guid);

  n = base +"/setType";
  pAveragingTypeCmd = new G4UIcmdWithAString(n,this);
  guid = G4String("Sets  averaging method ('DoseAveraged', 'TrackAveraged'). Default is 'DoseAveraged'.");
  pAveragingTypeCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableLETUncertaintyCmd) pLETActor->EnableLETUncertaintyImage(pEnableLETUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pSetLETtoWaterCmd) pLETActor->SetLETtoWater(pSetLETtoWaterCmd->GetNewBoolValue(newValue));
  if (cmd == pSetParallelCalculationCmd) pLETActor->SetParallelCalculation(pSetParallelCalculationCmd->GetNewBoolValue(newValue));

  if (cmd == pAveragingTypeCmd) pLETActor->SetLETType(newValue);

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATELETACTORMESSENGER_CC */
