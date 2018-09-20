/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateNeutronKermaActorMessenger.hh"
#include "GateNeutronKermaActor.hh"

//-----------------------------------------------------------------------------
GateNeutronKermaActorMessenger::GateNeutronKermaActorMessenger(GateNeutronKermaActor* sensor)
  :GateImageActorMessenger(sensor),
  pKermaActor(sensor)
{
  pEnableDoseCmd            = 0;
  pEnableDoseSquaredCmd     = 0;
  pEnableDoseUncertaintyCmd = 0;

  pEnableEdepCmd            = 0;
  pEnableEdepSquaredCmd     = 0;
  pEnableEdepUncertaintyCmd = 0;

  pEnableNumberOfHitsCmd    = 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNeutronKermaActorMessenger::~GateNeutronKermaActorMessenger()
{
  if(pEnableDoseCmd)            delete pEnableDoseCmd;
  if(pEnableDoseSquaredCmd)     delete pEnableDoseSquaredCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;

  if(pEnableEdepCmd)            delete pEnableEdepCmd;
  if(pEnableEdepSquaredCmd)     delete pEnableEdepSquaredCmd;
  if(pEnableEdepUncertaintyCmd) delete pEnableEdepUncertaintyCmd;

  if(pEnableNumberOfHitsCmd)    delete pEnableNumberOfHitsCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActorMessenger::BuildCommands(G4String base)
{
  pEnableDoseCmd = new G4UIcmdWithABool(G4String(base+"/enableDose"), this);
  pEnableDoseCmd->SetGuidance("Enable dose computation");

  pEnableDoseSquaredCmd = new G4UIcmdWithABool(G4String(base+"/enableSquaredDose"), this);
  pEnableDoseSquaredCmd->SetGuidance("Enable squared dose computation");

  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(G4String(base+"/enableUncertaintyDose"), this);
  pEnableDoseUncertaintyCmd->SetGuidance("Enable uncertainty dose computation");

  pEnableEdepCmd = new G4UIcmdWithABool(G4String(base+"/enableEdep"), this);
  pEnableEdepCmd->SetGuidance("Enable edep computation");

  pEnableEdepSquaredCmd = new G4UIcmdWithABool(G4String(base+"/enableSquaredEdep"), this);
  pEnableEdepSquaredCmd->SetGuidance("Enable squared edep computation");

  pEnableEdepUncertaintyCmd = new G4UIcmdWithABool(G4String(base+"/enableUncertaintyEdep"), this);
  pEnableEdepUncertaintyCmd->SetGuidance("Enable uncertainty edep computation");

  pEnableNumberOfHitsCmd = new G4UIcmdWithABool(G4String(base+"/enableNumberOfHits"), this);
  pEnableNumberOfHitsCmd->SetGuidance("Enable number of hits computation");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNeutronKermaActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd)            pKermaActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseSquaredCmd)     pKermaActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pKermaActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableEdepCmd)            pKermaActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepSquaredCmd)     pKermaActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepUncertaintyCmd) pKermaActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableNumberOfHitsCmd)    pKermaActor->EnableNumberOfHitsImage(pEnableNumberOfHitsCmd->GetNewBoolValue(newValue));

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------
