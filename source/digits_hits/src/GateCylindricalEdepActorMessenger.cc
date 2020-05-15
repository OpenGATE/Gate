/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATECYLINDRICALEDEPACTORMESSENGER_CC
#define GATECYLINDRICALEDEPACTORMESSENGER_CC

#include "GateCylindricalEdepActorMessenger.hh"
#include "GateCylindricalEdepActor.hh"

//-----------------------------------------------------------------------------
GateCylindricalEdepActorMessenger::GateCylindricalEdepActorMessenger(GateCylindricalEdepActor* sensor)
  :GateImageActorMessenger(sensor),
  pCylindicalEdepActor(sensor)
{

  pEnableDoseCmd = 0;
  pEnableEdepCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------
 

//-----------------------------------------------------------------------------
GateCylindricalEdepActorMessenger::~GateCylindricalEdepActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableEdepCmd) delete pEnableEdepCmd;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateCylindricalEdepActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);
  
  n = base+"/enableEdep";
  pEnableEdepCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep computation");
  pEnableEdepCmd->SetGuidance(guid);
  
  
  n = base+"/enableFluence";
  pEnableFluenceCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable fluence computation");
  pEnableFluenceCmd->SetGuidance(guid);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCylindricalEdepActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pCylindicalEdepActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pCylindicalEdepActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableFluenceCmd) pCylindicalEdepActor->EnableFluenceImage(pEnableFluenceCmd->GetNewBoolValue(newValue));
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif
