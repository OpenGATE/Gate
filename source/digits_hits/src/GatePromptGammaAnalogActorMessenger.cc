/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GatePromptGammaAnalogActorMessenger.hh"
#include "GatePromptGammaAnalogActor.hh"

//-----------------------------------------------------------------------------
GatePromptGammaAnalogActorMessenger::
GatePromptGammaAnalogActorMessenger(GatePromptGammaAnalogActor* v)
:GateImageActorMessenger(v), pPGAnalogActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaAnalogActorMessenger::~GatePromptGammaAnalogActorMessenger()
{
  DD("GatePromptGammaAnalogActorMessenger destructor");
  delete pSetInputDataFileCmd;
  delete pSetOutputCountCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/setInputDataFile";
  pSetInputDataFileCmd = new G4UIcmdWithAString(bb, this);
  G4String guidance = G4String("Set input root filename with proton/gamma energy 2D spectrum (obtained from PromptGammaStatisticsActor).");
  pSetInputDataFileCmd->SetGuidance(guidance);

  bb = base+"/setOutputCount";
  pSetOutputCountCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Set output to counts instead of yield.");
  pSetOutputCountCmd->SetGuidance(guidance);

  bb = base+"/setTimeNbBins";
  pTimeNbBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the time proton histograms");
  pTimeNbBinsCmd->SetGuidance(guidance);
  pTimeNbBinsCmd->SetParameterName("Nbins", false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaAnalogActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetInputDataFileCmd) pPGAnalogActor->SetInputDataFilename(newValue);
  if (cmd == pSetOutputCountCmd) pPGAnalogActor->SetOutputCount(pSetOutputCountCmd->GetNewBoolValue(newValue));
  if (cmd == pTimeNbBinsCmd) pPGAnalogActor->SetTimeNbBins(pTimeNbBinsCmd->GetNewIntValue(newValue));

  GateImageActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
