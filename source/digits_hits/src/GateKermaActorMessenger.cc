/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEKERMAACTORMESSENGER_CC
#define GATEKERMAACTORMESSENGER_CC

#include "GateKermaActorMessenger.hh"
#include "GateKermaActor.hh"

//-----------------------------------------------------------------------------
GateKermaActorMessenger::GateKermaActorMessenger(GateKermaActor* sensor)
  :GateImageActorMessenger(sensor),
  pKermaActor(sensor)
{

  pEnableDoseCmd = 0;
  pEnableDoseNormCmd= 0;
  pEnableDoseSquaredCmd= 0;
  pEnableDoseUncertaintyCmd= 0;
  pEnableDoseToWaterCmd = 0;
  pEnableDoseToWaterNormCmd= 0;
  pEnableDoseToWaterSquaredCmd= 0;
  pEnableDoseToWaterUncertaintyCmd= 0;
  pEnableEdepCmd= 0;
  pEnableEdepSquaredCmd= 0;
  pEnableEdepUncertaintyCmd= 0;
  pEnableNumberOfHitsCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateKermaActorMessenger::~GateKermaActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableDoseNormCmd) delete pEnableDoseNormCmd;
  if(pEnableDoseSquaredCmd) delete pEnableDoseSquaredCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;
  if(pEnableDoseToWaterCmd) delete pEnableDoseToWaterCmd;
  if(pEnableDoseToWaterNormCmd) delete pEnableDoseToWaterNormCmd;
  if(pEnableDoseToWaterSquaredCmd) delete pEnableDoseToWaterSquaredCmd;
  if(pEnableDoseToWaterUncertaintyCmd) delete pEnableDoseToWaterUncertaintyCmd;
  if(pEnableEdepCmd) delete pEnableEdepCmd;
  if(pEnableEdepSquaredCmd) delete pEnableEdepSquaredCmd;
  if(pEnableEdepUncertaintyCmd) delete pEnableEdepUncertaintyCmd;
  if(pEnableNumberOfHitsCmd) delete pEnableNumberOfHitsCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateKermaActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);

  n = base+"/normaliseDose";
  pEnableDoseNormCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose normalisation according to integral");
  pEnableDoseNormCmd->SetGuidance(guid);

  n = base+"/enableSquaredDose";
  pEnableDoseSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared dose computation");
  pEnableDoseSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyDose";
  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty dose computation");
  pEnableDoseUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableDoseToWater";
  pEnableDoseToWaterCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose to water computation");
  pEnableDoseToWaterCmd->SetGuidance(guid);

  n = base+"/normaliseDoseToWater";
  pEnableDoseToWaterNormCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose normalisation according to integral");
  pEnableDoseToWaterNormCmd->SetGuidance(guid);

  n = base+"/enableSquaredDoseToWater";
  pEnableDoseToWaterSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared dose to water computation");
  pEnableDoseToWaterSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyDoseToWater";
  pEnableDoseToWaterUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty dose to water computation");
  pEnableDoseToWaterUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableEdep";
  pEnableEdepCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep computation");
  pEnableEdepCmd->SetGuidance(guid);

  n = base+"/enableSquaredEdep";
  pEnableEdepSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared edep computation");
  pEnableEdepSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyEdep";
  pEnableEdepUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty edep computation");
  pEnableEdepUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableNumberOfHits";
  pEnableNumberOfHitsCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable number of hits computation");
  pEnableNumberOfHitsCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateKermaActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pKermaActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseSquaredCmd) pKermaActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pKermaActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterCmd) pKermaActor->EnableDoseToWaterImage(pEnableDoseToWaterCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterSquaredCmd) pKermaActor->EnableDoseToWaterSquaredImage(pEnableDoseToWaterSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterUncertaintyCmd) pKermaActor->EnableDoseToWaterUncertaintyImage(pEnableDoseToWaterUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pKermaActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepSquaredCmd) pKermaActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepUncertaintyCmd) pKermaActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableNumberOfHitsCmd) pKermaActor->EnableNumberOfHitsImage(pEnableNumberOfHitsCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseNormCmd) pKermaActor->EnableDoseNormalisation(pEnableDoseNormCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterNormCmd) pKermaActor->EnableDoseToWaterNormalisation(pEnableDoseToWaterNormCmd->GetNewBoolValue(newValue));

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEDOSEACTORMESSENGER_CC */
