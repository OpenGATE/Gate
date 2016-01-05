/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEDOSEACTORMESSENGER_CC
#define GATEDOSEACTORMESSENGER_CC

#include "GateDoseActorMessenger.hh"
#include "GateDoseActor.hh"

//-----------------------------------------------------------------------------
GateDoseActorMessenger::GateDoseActorMessenger(GateDoseActor* sensor)
  :GateImageActorMessenger(sensor),
  pDoseActor(sensor)
{

  pEnableDoseCmd = 0;
  pEnableDoseNormToMaxCmd= 0;
  pEnableDoseNormToIntegralCmd= 0;
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
  pSetDoseAlgorithmCmd= 0;
  pImportMassImageCmd= 0;
  pExportMassImageCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDoseActorMessenger::~GateDoseActorMessenger()
{
  if(pEnableDoseCmd) delete pEnableDoseCmd;
  if(pEnableDoseNormToMaxCmd) delete pEnableDoseNormToMaxCmd;
  if(pEnableDoseNormToIntegralCmd) delete pEnableDoseNormToIntegralCmd;
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
  if(pSetDoseAlgorithmCmd) delete pSetDoseAlgorithmCmd;
  if(pImportMassImageCmd) delete pImportMassImageCmd;
  if(pExportMassImageCmd) delete pExportMassImageCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);

  n = base+"/normaliseDoseToMax";
  pEnableDoseNormToMaxCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose normalisation according to max");
  pEnableDoseNormToMaxCmd->SetGuidance(guid);

  n = base+"/normaliseDoseToIntegral";
  pEnableDoseNormToIntegralCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose normalisation according to integral");
  pEnableDoseNormToIntegralCmd->SetGuidance(guid);

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

  n = base+"/exportMassImage";
  pExportMassImageCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Export mass image");
  pExportMassImageCmd->SetGuidance(guid);
  pExportMassImageCmd->SetParameterName("Export mass image",false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseSquaredCmd) pDoseActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd) pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterCmd) pDoseActor->EnableDoseToWaterImage(pEnableDoseToWaterCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterSquaredCmd) pDoseActor->EnableDoseToWaterSquaredImage(pEnableDoseToWaterSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterUncertaintyCmd) pDoseActor->EnableDoseToWaterUncertaintyImage(pEnableDoseToWaterUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pDoseActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepSquaredCmd) pDoseActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepUncertaintyCmd) pDoseActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableNumberOfHitsCmd) pDoseActor->EnableNumberOfHitsImage(pEnableNumberOfHitsCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableDoseNormToMaxCmd) pDoseActor->EnableDoseNormalisationToMax(pEnableDoseNormToMaxCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseNormToIntegralCmd) pDoseActor->EnableDoseNormalisationToIntegral(pEnableDoseNormToIntegralCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseToWaterNormCmd) pDoseActor->EnableDoseToWaterNormalisation(pEnableDoseToWaterNormCmd->GetNewBoolValue(newValue));

  if (cmd == pSetDoseAlgorithmCmd) pDoseActor->SetDoseAlogithm(newValue);
  if (cmd == pImportMassImageCmd) pDoseActor->ImportMassImage(newValue);
  if (cmd == pExportMassImageCmd) pDoseActor->ExportMassImage(newValue);

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif 
