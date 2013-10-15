/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEHYBRIDDOSEACTORMESSENGER_CC
#define GATEHYBRIDDOSEACTORMESSENGER_CC

#include "GateHybridDoseActorMessenger.hh"
#include "GateHybridDoseActor.hh"

//-----------------------------------------------------------------------------
GateHybridDoseActorMessenger::GateHybridDoseActorMessenger(GateHybridDoseActor* sensor)
  :GateImageActorMessenger(sensor),
   pDoseActor(sensor)
{
  pEnableDoseCmd = 0;
  pEnableEdepCmd= 0;
  pEnableDoseUncertaintyCmd= 0;
  pSetPrimaryMultiplicityCmd = 0;
  pSetSecondaryMultiplicityCmd = 0;
  pSetSecondaryMultiplicityCmd2 = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateHybridDoseActorMessenger::~GateHybridDoseActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableEdepCmd) delete pEnableEdepCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;
  if(pSetPrimaryMultiplicityCmd) delete pSetPrimaryMultiplicityCmd;
  if(pSetSecondaryMultiplicityCmd) delete pSetSecondaryMultiplicityCmd;
  if(pSetSecondaryMultiplicityCmd2) delete pSetSecondaryMultiplicityCmd2;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridDoseActorMessenger::BuildCommands(G4String base)
{
//   GateImageActorMessenger::BuildCommands(base);
  
  G4String n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this); 
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);
  
  n = base+"/enableEdep";
  pEnableEdepCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable edep computation");
  pEnableEdepCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyDose";
  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this); 
  guid = G4String("Enable uncertainty dose computation");
  pEnableDoseUncertaintyCmd->SetGuidance(guid);
  
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
void GateHybridDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  GateImageActorMessenger::SetNewValue( cmd, newValue);
  
   // if (cmd == pEnableDoseCmd) pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pDoseActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
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
