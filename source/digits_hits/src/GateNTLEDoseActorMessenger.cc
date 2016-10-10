/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATENTLEDOSEACTORMESSENGER_CC
#define GATENTLEDOSEACTORMESSENGER_CC

#include "GateNTLEDoseActorMessenger.hh"
#include "GateNTLEDoseActor.hh"

//-----------------------------------------------------------------------------
GateNTLEDoseActorMessenger::GateNTLEDoseActorMessenger(GateNTLEDoseActor* sensor)
  :GateImageActorMessenger(sensor),
   pDoseActor(sensor)
{
  pEnableDoseCmd = 0;
  pEnableDoseSquaredCmd= 0;
  pEnableDoseUncertaintyCmd= 0;
  pEnableDoseCorrectionCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNTLEDoseActorMessenger::~GateNTLEDoseActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableDoseSquaredCmd) delete pEnableDoseSquaredCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;
  if(pEnableDoseCorrectionCmd) delete pEnableDoseCorrectionCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);

  n = base+"/enableSquaredDose";
  pEnableDoseSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared dose computation");
  pEnableDoseSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyDose";
  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty dose computation");
  pEnableDoseUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableDoseCorrection";
  pEnableDoseCorrectionCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose correction computation");
  pEnableDoseCorrectionCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseSquaredCmd) pDoseActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseCorrectionCmd) pDoseActor->EnableDoseCorrection(pEnableDoseCorrectionCmd->GetNewBoolValue(newValue));

 GateImageActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif
