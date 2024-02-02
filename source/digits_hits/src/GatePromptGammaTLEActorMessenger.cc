/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaTLEActor.hh"
#include "GatePromptGammaTLEActorMessenger.hh"

//-----------------------------------------------------------------------------
GatePromptGammaTLEActorMessenger::
GatePromptGammaTLEActorMessenger(GatePromptGammaTLEActor* v)
:GateImageActorMessenger(v), pPGTLEActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaTLEActorMessenger::~GatePromptGammaTLEActorMessenger()
{
  DD("GatePromptGammaTLEActorMessenger destructor");
  delete pSetInputDataFileCmd;
  delete pEnableDebugOutputCmd;
  //delete pEnableSysVarianceCmd;
  //delete pEnableIntermediaryUncertaintyOutputCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/setInputDataFile";
  pSetInputDataFileCmd = new G4UIcmdWithAString(bb, this);
  G4String guidance = G4String("Set input root filename with proton/gamma energy 2D spectrum (obtained from PromptGammaStatisticsActor).");
  pSetInputDataFileCmd->SetGuidance(guidance);

  bb = base+"/enableDebugOutput";
  pEnableDebugOutputCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable tracklengths output (L and L^2) and Gamma_M (per voxel per E_gamma). May be used to compute variance afterwards.");
  pEnableDebugOutputCmd->SetGuidance(guidance);

  bb = base+"/enableOutputMatch";
  pEnableOutputMatchCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Enable this too make sure the regular TLE output and debug output match. In corner cases where voxels of the image and TLE actor don't match, DebugOutput will take the material at the voxel center, while regular TLE will take the material at the interaction point. Enabling this will force regular TLE to also look at the voxel center.");
  pEnableOutputMatchCmd->SetGuidance(guidance);

  bb = base+"/setTimeNbBins";
  pTimeNbBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the time proton histograms");
  pTimeNbBinsCmd->SetGuidance(guidance);
  pTimeNbBinsCmd->SetParameterName("Nbins", false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaTLEActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetInputDataFileCmd) pPGTLEActor->SetInputDataFilename(newValue);
  if (cmd == pEnableDebugOutputCmd) pPGTLEActor->EnableDebugOutput(pEnableDebugOutputCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableOutputMatchCmd) pPGTLEActor->EnableOutputMatch(pEnableOutputMatchCmd->GetNewBoolValue(newValue));
  if (cmd == pTimeNbBinsCmd) pPGTLEActor->SetTimeNbBins(pTimeNbBinsCmd->GetNewIntValue(newValue));
  //if (cmd == pEnableSysVarianceCmd) pPGTLEActor->EnableSysVarianceImage(pEnableSysVarianceCmd->GetNewBoolValue(newValue));
  //if (cmd == pEnableIntermediaryUncertaintyOutputCmd) pPGTLEActor->EnableIntermediaryUncertaintyOutput(pEnableIntermediaryUncertaintyOutputCmd->GetNewBoolValue(newValue));
  GateImageActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
