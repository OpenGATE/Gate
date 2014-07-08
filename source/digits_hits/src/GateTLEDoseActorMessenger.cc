/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATETLEDOSEACTORMESSENGER_CC
#define GATETLEDOSEACTORMESSENGER_CC

#include "GateTLEDoseActorMessenger.hh"

#include "GateTLEDoseActor.hh"

//-----------------------------------------------------------------------------
GateTLEDoseActorMessenger::GateTLEDoseActorMessenger(GateTLEDoseActor* sensor)
  :GateImageActorMessenger(sensor),
   pDoseActor(sensor)
{
  pEnableDoseCmd = 0;
  pEnableEdepCmd= 0;
  pEnableDoseUncertaintyCmd= 0;
  pEnableDoseSquaredCmd= 0;
  pEnableEdepSquaredCmd= 0;
  pEnableEdepUncertaintyCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTLEDoseActorMessenger::~GateTLEDoseActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableEdepCmd) delete pEnableEdepCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;
  if(pEnableDoseSquaredCmd) delete pEnableDoseSquaredCmd;
  if(pEnableEdepSquaredCmd) delete pEnableEdepSquaredCmd;
  if(pEnableEdepUncertaintyCmd) delete pEnableEdepUncertaintyCmd;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);

  n = base+"/enableEdep";
  pEnableEdepCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep computation");
  pEnableEdepCmd->SetGuidance(guid);

  n = base+"/enableSquaredDose";
  pEnableDoseSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared dose computation");
  pEnableDoseSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyDose";
  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty dose computation");
  pEnableDoseUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableSquaredEdep";
  pEnableEdepSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared edep computation");
  pEnableEdepSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyEdep";
  pEnableEdepUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty edep computation");
  pEnableEdepUncertaintyCmd->SetGuidance(guid);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pDoseActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
 if (cmd == pEnableDoseSquaredCmd) pDoseActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
 if (cmd == pEnableEdepSquaredCmd) pDoseActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
 if (cmd == pEnableEdepUncertaintyCmd) pDoseActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));

 GateImageActorMessenger::SetNewValue( cmd, newValue);

}
//-----------------------------------------------------------------------------

#endif /* end #define GATETLEDOSEACTORMESSENGER_CC */
