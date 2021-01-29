/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  pSetLETtoWaterCmd = 0;
  pAveragingTypeCmd = 0;
  pSetParallelCalculationCmd = 0;
  pSetOtherMaterialCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateLETActorMessenger::~GateLETActorMessenger()
{
  if(pSetLETtoWaterCmd) delete pSetLETtoWaterCmd;
  if(pAveragingTypeCmd) delete pAveragingTypeCmd;
  if(pSetParallelCalculationCmd) delete pSetParallelCalculationCmd;
  if(pSetOtherMaterialCmd) delete pSetOtherMaterialCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/setLETtoWater";
  pSetLETtoWaterCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose-to-water correction in LET calculation");
  pSetLETtoWaterCmd->SetGuidance(guid);

  n = base+"/doParallelCalculation";
  pSetParallelCalculationCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable parallel calculation: creates 2 output files for each simulation");
  pSetParallelCalculationCmd->SetGuidance(guid);

  n = base +"/setType";
  pAveragingTypeCmd = new G4UIcmdWithAString(n,this);
  guid = G4String("Sets  averaging method ('DoseAveraged', 'TrackAveraged'). Default is 'DoseAveraged'.");
  pAveragingTypeCmd->SetGuidance(guid);
  
  n = base+"/setOtherMaterial";
  pSetOtherMaterialCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Set Other Material Name");
  pSetOtherMaterialCmd->SetGuidance(guid);

  n = base+"/setCutVal";
  pCutValCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Set cut value for restricted LET");
  pCutValCmd->SetGuidance(guid);
  pCutValCmd->SetParameterName("CutValue", false);
  pCutValCmd->SetDefaultUnit("MeV");
  
  n = base+"/setLETthresholdMin";
  pThrMinCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Set cut value for restricted LET");
  pThrMinCmd->SetGuidance(guid);
  pThrMinCmd->SetParameterName("LETthresholdMin", false);
  pThrMinCmd->SetDefaultUnit("MeV/mm");
  
  n = base+"/setLETthresholdMax";
  pThrMaxCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Set cut value for restricted LET");
  pThrMaxCmd->SetGuidance(guid);
  pThrMaxCmd->SetParameterName("LETthresholdMax", false);
  pThrMaxCmd->SetDefaultUnit("MeV/mm");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateLETActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetLETtoWaterCmd) pLETActor->SetLETtoWater(pSetLETtoWaterCmd->GetNewBoolValue(newValue));
  if (cmd == pSetParallelCalculationCmd) pLETActor->SetParallelCalculation(pSetParallelCalculationCmd->GetNewBoolValue(newValue));

  if (cmd == pAveragingTypeCmd) pLETActor->SetLETType(newValue);
  if (cmd == pSetOtherMaterialCmd) pLETActor->SetMaterial(newValue);
  if (cmd == pCutValCmd) pLETActor->SetCutVal(pCutValCmd->GetNewDoubleValue(newValue));
  if (cmd == pThrMinCmd) pLETActor->SetLETthrMin(pThrMinCmd->GetNewDoubleValue(newValue));
  if (cmd == pThrMaxCmd) pLETActor->SetLETthrMax(pThrMaxCmd->GetNewDoubleValue(newValue));

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATELETACTORMESSENGER_CC */
