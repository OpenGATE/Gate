/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateFluenceActorMessenger.hh"
#include "GateFluenceActor.hh"

//-----------------------------------------------------------------------------
GateFluenceActorMessenger::GateFluenceActorMessenger(GateFluenceActor* sensor) :
    GateImageActorMessenger(sensor), pFluenceActor(sensor)
  {
  //******************************************************************************************
  pEnableSquaredCmd = 0;
  pEnableUncertaintyCmd = 0;
  pEnableNormCmd = 0;
  //pEnableNumberOfHitsCmd= 0;
  //******************************************************************************************

  pEnableScatterCmd = 0;
  BuildCommands(baseName + sensor->GetObjectName());
  }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateFluenceActorMessenger::~GateFluenceActorMessenger()
  {
  //*********************************************************************************
  if (pEnableSquaredCmd)
    delete pEnableSquaredCmd;
  if (pEnableUncertaintyCmd)
    delete pEnableUncertaintyCmd;
  if (pEnableNormCmd)
    delete pEnableNormCmd;
  //if(pEnableNumberOfHitsCmd) delete pEnableNumberOfHitsCmd;
  //*********************************************************************************

  if (pEnableScatterCmd)
    delete pEnableScatterCmd;
  }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFluenceActorMessenger::BuildCommands(G4String base)
  {
  G4String n = base + "/enableScatter";
  pEnableScatterCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable computation of scattered particles fluence");
  pEnableScatterCmd->SetGuidance(guid);

  n = base + "/enableSquared";
  pEnableSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared image statistic");
  pEnableSquaredCmd->SetGuidance(guid);

  n = base + "/enableUncertainty";
  pEnableUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty image statistic");
  pEnableUncertaintyCmd->SetGuidance(guid);

  n = base + "/enableNormalise";
  pEnableNormCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable normalise image");
  pEnableNormCmd->SetGuidance(guid);

  n = base + "/enableNumberOfHits";
  pEnableNumberOfHitsCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable number of hits computation");
  pEnableNumberOfHitsCmd->SetGuidance(guid);

  n = base + "/ignoreWeight";
  pSetIgnoreWeightCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable weight of particles");
  pSetIgnoreWeightCmd->SetGuidance(guid);

  n = base + "/responseDetectorFilename";
  pSetResponseDetectorFileCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Response detector curve (weight to each energy)");
  pSetResponseDetectorFileCmd->SetGuidance(guid);

  n = base + "/scatterOrderFilename";
  pSetScatterOrderFilenameCmd = new G4UIcmdWithAString(n, this);
  guid = "Set the file name for the scatter x-rays that hit the detector (printf format with runId as a single parameter).";
  pSetScatterOrderFilenameCmd->SetGuidance(guid);

  n = base + "/separateProcessFilename";
  pSetSeparateProcessFilenameCmd = new G4UIcmdWithAString(n, this);
  guid = "Set the file name for the different scatter x-rays processes that hit the detector (printf format with runId as a single parameter).";
  pSetSeparateProcessFilenameCmd->SetGuidance(guid);
  }
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFluenceActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
  {
  if (cmd == pEnableSquaredCmd)
    {
    pFluenceActor->EnableSquaredImage(pEnableSquaredCmd->GetNewBoolValue(newValue));
    }
  if (cmd == pEnableUncertaintyCmd)
    {
    pFluenceActor->EnableUncertaintyImage(pEnableUncertaintyCmd->GetNewBoolValue(newValue));
    }
  if (cmd == pEnableNormCmd)
    {
    pFluenceActor->EnableNormalisation(pEnableNormCmd->GetNewBoolValue(newValue));
    }
  if (cmd == pEnableNumberOfHitsCmd)
    {
    pFluenceActor->EnableNumberOfHitsImage(pEnableNumberOfHitsCmd->GetNewBoolValue(newValue));
    }
  if (cmd == pSetIgnoreWeightCmd)
    {
    pFluenceActor->SetIgnoreWeight(pSetIgnoreWeightCmd->GetNewBoolValue(newValue));
    }
  if (cmd == pEnableScatterCmd)
    {
    pFluenceActor->EnableScatterImage(pEnableScatterCmd->GetNewBoolValue(newValue));
    }
  if (cmd == pSetResponseDetectorFileCmd)
    {
    pFluenceActor->SetResponseDetectorFile(newValue);
    }
  if (cmd == pSetScatterOrderFilenameCmd)
    {
    pFluenceActor->SetScatterOrderFilename(newValue);
    }
  if (cmd == pSetSeparateProcessFilenameCmd)
    {
    pFluenceActor->SetSeparateProcessFilename(newValue);
    }

  GateImageActorMessenger::SetNewValue(cmd, newValue);
  }
//-----------------------------------------------------------------------------
