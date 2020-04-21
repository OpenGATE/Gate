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
  pEnableDoseNormToMaxCmd= 0;
  pEnableDoseNormToIntegralCmd= 0;
  pEnableDoseSquaredCmd= 0;
  pEnableDoseUncertaintyCmd= 0;
  pEnableDoseToWaterCmd = 0;
  pEnableDoseToWaterNormCmd= 0;
  pEnableDoseToWaterSquaredCmd= 0;
  pEnableDoseToWaterUncertaintyCmd= 0;
  pEnableEdepCmd= 0;
  
  pEnableEdepHadElasticCmd= 0;
  pEnableEdepInelasticCmd= 0;
  pEnableEdepRestCmd= 0;
  
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
GateCylindricalEdepActorMessenger::~GateCylindricalEdepActorMessenger()
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
  if(pEnableEdepHadElasticCmd) delete pEnableEdepHadElasticCmd;
  if(pEnableEdepInelasticCmd) delete pEnableEdepInelasticCmd;
  if(pEnableEdepRestCmd) delete pEnableEdepRestCmd;
  
  
  if(pEnableEdepSquaredCmd) delete pEnableEdepSquaredCmd;
  if(pEnableEdepUncertaintyCmd) delete pEnableEdepUncertaintyCmd;
  if(pEnableNumberOfHitsCmd) delete pEnableNumberOfHitsCmd;
  if(pSetDoseAlgorithmCmd) delete pSetDoseAlgorithmCmd;
  if(pImportMassImageCmd) delete pImportMassImageCmd;
  if(pExportMassImageCmd) delete pExportMassImageCmd;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateCylindricalEdepActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);
  

  //n = base+"/enableSquaredDose";
  //pEnableDoseSquaredCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable squared dose computation");
  //pEnableDoseSquaredCmd->SetGuidance(guid);

  //n = base+"/enableUncertaintyDose";
  //pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable uncertainty dose computation");
  //pEnableDoseUncertaintyCmd->SetGuidance(guid);

  //n = base+"/enableDoseToWater";
  //pEnableDoseToWaterCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable dose to water computation");
  //pEnableDoseToWaterCmd->SetGuidance(guid);

  //n = base+"/normaliseDoseToWater";
  //pEnableDoseToWaterNormCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable dose normalisation according to integral");
  //pEnableDoseToWaterNormCmd->SetGuidance(guid);

  //n = base+"/enableSquaredDoseToWater";
  //pEnableDoseToWaterSquaredCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable squared dose to water computation");
  //pEnableDoseToWaterSquaredCmd->SetGuidance(guid);

  //n = base+"/enableUncertaintyDoseToWater";
  //pEnableDoseToWaterUncertaintyCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable uncertainty dose to water computation");
  //pEnableDoseToWaterUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableEdep";
  pEnableEdepCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep computation");
  pEnableEdepCmd->SetGuidance(guid);

  n = base+"/enableEdepHadElastic";
  pEnableEdepHadElasticCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep had elastic computation");
  pEnableEdepHadElasticCmd->SetGuidance(guid);
  
  n = base+"/enableEdepInelastic";
  pEnableEdepInelasticCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep computation");
  pEnableEdepInelasticCmd->SetGuidance(guid);

  n = base+"/enableEdepRest";
  pEnableEdepRestCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable edep computation");
  pEnableEdepRestCmd->SetGuidance(guid);


  //n = base+"/enableSquaredEdep";
  //pEnableEdepSquaredCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable squared edep computation");
  //pEnableEdepSquaredCmd->SetGuidance(guid);

  //n = base+"/enableUncertaintyEdep";
  //pEnableEdepUncertaintyCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable uncertainty edep computation");
  //pEnableEdepUncertaintyCmd->SetGuidance(guid);

  //n = base+"/enableNumberOfHits";
  //pEnableNumberOfHitsCmd = new G4UIcmdWithABool(n, this);
  //guid = G4String("Enable number of hits computation");
  //pEnableNumberOfHitsCmd->SetGuidance(guid);

  //n = base+"/setDoseAlgorithm";
  //pSetDoseAlgorithmCmd = new G4UIcmdWithAString(n, this);
  //guid = G4String("Set the alogrithm used in the dose calculation");
  //pSetDoseAlgorithmCmd->SetGuidance(guid);
  //pSetDoseAlgorithmCmd->SetParameterName("Dose algorithm",false);

  //n = base+"/importMassImage";
  //pImportMassImageCmd = new G4UIcmdWithAString(n, this);
  //guid = G4String("Import mass image");
  //pImportMassImageCmd->SetGuidance(guid);
  //pImportMassImageCmd->SetParameterName("Import mass image",false);

  //n = base+"/exportMassImage";
  //pExportMassImageCmd = new G4UIcmdWithAString(n, this);
  //guid = G4String("Export mass image");
  //pExportMassImageCmd->SetGuidance(guid);
  //pExportMassImageCmd->SetParameterName("Export mass image",false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCylindricalEdepActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableDoseCmd) pCylindicalEdepActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseSquaredCmd) pCylindicalEdepActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseUncertaintyCmd) pCylindicalEdepActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseToWaterCmd) pCylindicalEdepActor->EnableDoseToWaterImage(pEnableDoseToWaterCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseToWaterSquaredCmd) pCylindicalEdepActor->EnableDoseToWaterSquaredImage(pEnableDoseToWaterSquaredCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseToWaterUncertaintyCmd) pCylindicalEdepActor->EnableDoseToWaterUncertaintyImage(pEnableDoseToWaterUncertaintyCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepCmd) pCylindicalEdepActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
    if (cmd == pEnableEdepHadElasticCmd) pCylindicalEdepActor->EnableEdepHadElasticImage(pEnableEdepHadElasticCmd->GetNewBoolValue(newValue));
      if (cmd == pEnableEdepInelasticCmd) pCylindicalEdepActor->EnableEdepInelasticImage(pEnableEdepInelasticCmd->GetNewBoolValue(newValue));
       if (cmd == pEnableEdepRestCmd) pCylindicalEdepActor->EnableEdepRestImage(pEnableEdepRestCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableEdepSquaredCmd) pCylindicalEdepActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableEdepUncertaintyCmd) pCylindicalEdepActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableNumberOfHitsCmd) pCylindicalEdepActor->EnableNumberOfHitsImage(pEnableNumberOfHitsCmd->GetNewBoolValue(newValue));

  //if (cmd == pEnableDoseNormToMaxCmd) pCylindicalEdepActor->EnableDoseNormalisationToMax(pEnableDoseNormToMaxCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseNormToIntegralCmd) pCylindicalEdepActor->EnableDoseNormalisationToIntegral(pEnableDoseNormToIntegralCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableDoseToWaterNormCmd) pCylindicalEdepActor->EnableDoseToWaterNormalisation(pEnableDoseToWaterNormCmd->GetNewBoolValue(newValue));

  //if (cmd == pSetDoseAlgorithmCmd) pCylindicalEdepActor->SetDoseAlgorithmType(newValue);
  //if (cmd == pImportMassImageCmd) pCylindicalEdepActor->ImportMassImage(newValue);
  //if (cmd == pExportMassImageCmd) pCylindicalEdepActor->ExportMassImage(newValue);

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif
