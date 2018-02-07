/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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
  pEnableDoseSquaredCmd= 0;
  pEnableDoseUncertaintyCmd= 0;
  pEnableEdepCmd= 0;
  pEnableEdepSquaredCmd= 0;
  pEnableEdepUncertaintyCmd= 0;
  pEnableDoseNormToMaxCmd= 0;
  pEnableDoseNormToIntegralCmd= 0;
  pSetDoseAlgorithmCmd= 0;
  pImportMassImageCmd= 0;
  pVolumeFilterCmd= 0;
  pMaterialFilterCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTLEDoseActorMessenger::~GateTLEDoseActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableDoseSquaredCmd) delete pEnableDoseSquaredCmd;
  if(pEnableDoseUncertaintyCmd) delete pEnableDoseUncertaintyCmd;
  if(pEnableEdepCmd) delete pEnableEdepCmd;
  if(pEnableEdepSquaredCmd) delete pEnableEdepSquaredCmd;
  if(pEnableEdepUncertaintyCmd) delete pEnableEdepUncertaintyCmd;
  if(pEnableDoseNormToMaxCmd) delete pEnableDoseNormToMaxCmd;
  if(pEnableDoseNormToIntegralCmd) delete pEnableDoseNormToIntegralCmd;
  if(pSetDoseAlgorithmCmd) delete pSetDoseAlgorithmCmd;
  if(pImportMassImageCmd) delete pImportMassImageCmd;
  if(pVolumeFilterCmd) delete pVolumeFilterCmd;
  if(pMaterialFilterCmd) delete pMaterialFilterCmd;
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

  n = base+"/normaliseDoseToMax";
  pEnableDoseNormToMaxCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose normalisation according to max");
  pEnableDoseNormToMaxCmd->SetGuidance(guid);

  n = base+"/normaliseDoseToIntegral";
  pEnableDoseNormToIntegralCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose normalisation according to integral");
  pEnableDoseNormToIntegralCmd->SetGuidance(guid);

  n = base+"/setDoseAlgorithm";
  pSetDoseAlgorithmCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Set the alogrithm used in the dose calculation");
  pSetDoseAlgorithmCmd->SetGuidance(guid);
  pSetDoseAlgorithmCmd->SetParameterName("Dose algorithm",false);

  n = base+"/importMassImage";
  pImportMassImageCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Import mass image");
  pImportMassImageCmd->SetGuidance(guid);
  pImportMassImageCmd->SetParameterName("Import mass image",false);

  n = base+"/setVolumeFilter";
  pVolumeFilterCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Volume filter");
  pVolumeFilterCmd->SetGuidance(guid);
  pVolumeFilterCmd->SetParameterName("Volume filter",false);

  n = base+"/setMaterialFilter";
  pMaterialFilterCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Material filter");
  pMaterialFilterCmd->SetGuidance(guid);
  pMaterialFilterCmd->SetParameterName("Material filter",false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLEDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseSquaredCmd) pDoseActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pDoseActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepSquaredCmd) pDoseActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepUncertaintyCmd) pDoseActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseNormToMaxCmd) pDoseActor->EnableDoseNormalisationToMax(pEnableDoseNormToMaxCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseNormToIntegralCmd) pDoseActor->EnableDoseNormalisationToIntegral(pEnableDoseNormToIntegralCmd->GetNewBoolValue(newValue));
  if (cmd == pSetDoseAlgorithmCmd) pDoseActor->SetDoseAlgorithmType(newValue);
  if (cmd == pImportMassImageCmd) pDoseActor->ImportMassImage(newValue);
  if (cmd == pVolumeFilterCmd) pDoseActor->VolumeFilter(newValue);
  if (cmd == pMaterialFilterCmd) pDoseActor->MaterialFilter(newValue);

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif
