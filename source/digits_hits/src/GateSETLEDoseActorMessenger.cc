/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GATESETLEDOSEACTORMESSENGER_CC
#define GATESETLEDOSEACTORMESSENGER_CC

#include "GateSETLEDoseActorMessenger.hh"
#include "GateSETLEDoseActor.hh"

//-----------------------------------------------------------------------------
GateSETLEDoseActorMessenger::GateSETLEDoseActorMessenger(GateSETLEDoseActor* sensor)
  :GateImageActorMessenger(sensor),
   pDoseActor(sensor)
{
  pEnableDoseCmd = 0;
  pEnableDoseUncertaintyCmd= 0;
  pEnablePrimaryDoseCmd = 0;
  pEnablePrimaryDoseUncertaintyCmd= 0;
  pEnableSecondaryDoseCmd = 0;
  pEnableSecondaryDoseUncertaintyCmd= 0;
  pEnableHybridinoCmd= 0;
  pSetPrimaryMultiplicityCmd = 0;
  pSetSecondaryMultiplicityCmd = 0;
  pSetSecondaryMultiplicityCmd2 = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSETLEDoseActorMessenger::~GateSETLEDoseActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;
  if(pEnablePrimaryDoseCmd) delete pEnablePrimaryDoseCmd;
  if(pEnablePrimaryDoseUncertaintyCmd) delete pEnablePrimaryDoseUncertaintyCmd;
  if(pEnableSecondaryDoseCmd) delete pEnableSecondaryDoseCmd;
  if(pEnableSecondaryDoseUncertaintyCmd) delete pEnableSecondaryDoseUncertaintyCmd;  
  if(pEnableHybridinoCmd) delete pEnableHybridinoCmd;

  if(pSetPrimaryMultiplicityCmd) delete pSetPrimaryMultiplicityCmd;
  if(pSetSecondaryMultiplicityCmd) delete pSetSecondaryMultiplicityCmd;
  if(pSetSecondaryMultiplicityCmd2) delete pSetSecondaryMultiplicityCmd2;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSETLEDoseActorMessenger::BuildCommands(G4String base)
{
//   GateImageActorMessenger::BuildCommands(base);
  
  G4String n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this); 
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);
  
  n = base+"/enableUncertaintyDose";
  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable uncertainty dose computation");
  pEnableDoseUncertaintyCmd->SetGuidance(guid);

  n = base+"/enablePrimaryDose";
  pEnablePrimaryDoseCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable primary dose computation");
  pEnablePrimaryDoseCmd->SetGuidance(guid);
  
  n = base+"/enablePrimaryUncertaintyDose";
  pEnablePrimaryDoseUncertaintyCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable primary uncertainty dose computation");
  pEnablePrimaryDoseUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableSecondaryDose";
  pEnableSecondaryDoseCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable secondary dose computation");
  pEnableSecondaryDoseCmd->SetGuidance(guid);
  
  n = base+"/enableSecondaryUncertaintyDose";
  pEnableSecondaryDoseUncertaintyCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable secondary uncertainty dose computation");
  pEnableSecondaryDoseUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableHybridino";
  pEnableHybridinoCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable hybrid particle navigation (raycasting otherwise)");
  pEnableHybridinoCmd->SetGuidance(guid);

  n = base+"/setPrimaryMultiplicity";
  pSetPrimaryMultiplicityCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of hybrid particle by primary particle generated");
  pSetPrimaryMultiplicityCmd->SetGuidance(guid);

  n = base+"/setSecondaryMultiplicity";
  pSetSecondaryMultiplicityCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of hybrid particle by secondary particle generated");
  pSetSecondaryMultiplicityCmd->SetGuidance(guid);
  
  n = base+"/setSecondaryMultiplicityUsingTime";
  pSetSecondaryMultiplicityCmd2 = new GateUIcmdWithTwoDouble(n, this); 
  guid = G4String("Set the number of hybrid particle by secondary particle generated using time and primary particle number");
  pSetSecondaryMultiplicityCmd2->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSETLEDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  GateImageActorMessenger::SetNewValue( cmd, newValue);
  
  if (cmd == pEnableDoseCmd) pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnablePrimaryDoseCmd) pDoseActor->EnablePrimaryDoseImage(pEnablePrimaryDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnablePrimaryDoseUncertaintyCmd) pDoseActor->EnablePrimaryDoseUncertaintyImage(pEnablePrimaryDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableSecondaryDoseCmd) pDoseActor->EnableSecondaryDoseImage(pEnableSecondaryDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableHybridinoCmd) pDoseActor->EnableHybridino(pEnableHybridinoCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableSecondaryDoseUncertaintyCmd) pDoseActor->EnableSecondaryDoseUncertaintyImage(pEnableSecondaryDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pSetPrimaryMultiplicityCmd) pDoseActor->SetPrimaryMultiplicity(pSetPrimaryMultiplicityCmd->GetNewIntValue(newValue));
  if (cmd == pSetSecondaryMultiplicityCmd) pDoseActor->SetSecondaryMultiplicity(pSetSecondaryMultiplicityCmd->GetNewIntValue(newValue));
  if (cmd == pSetSecondaryMultiplicityCmd2)
  {
    G4double t = pSetSecondaryMultiplicityCmd2->GetNewDoubleValue(0,newValue);
    G4double n = pSetSecondaryMultiplicityCmd2->GetNewDoubleValue(1,newValue);
    pDoseActor->SetSecondaryMultiplicity(t,n);   
  }
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEHYBRIDDOSEACTORMESSENGER_CC */
