/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSigmoidalThresholderMessenger.hh"

#include "GateSigmoidalThresholder.hh"

#include "G4UIcmdWithADouble.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateSigmoidalThresholderMessenger::GateSigmoidalThresholderMessenger(GateSigmoidalThresholder* itsSigThresholder)
    : GatePulseProcessorMessenger(itsSigThresholder)
{
  G4String guidance;
  G4String cmdName1;
  G4String cmdName2;
  G4String cmdName3;

  cmdName1 = GetDirectoryName() + "setThreshold";
  thresholdCmd = new G4UIcmdWithADoubleAndUnit(cmdName1,this);
  thresholdCmd->SetGuidance("Set threshold (in keV) for sigmoidal pulse-discrimination");

  cmdName2 = GetDirectoryName() + "setThresholdAlpha";
  alphaCmd = new G4UIcmdWithADouble(cmdName2,this);
  alphaCmd->SetGuidance("Set the alpha parameter of the sigmoidal function");

  cmdName3 = GetDirectoryName() + "setThresholdPerCent";
  perCentCmd = new G4UIcmdWithADouble(cmdName3,this);
  perCentCmd->SetGuidance("Set the per cent of acceptance for sigmoidal threshold pulse-discrimination");
}


GateSigmoidalThresholderMessenger::~GateSigmoidalThresholderMessenger()
{
  delete thresholdCmd;
  delete alphaCmd;
  delete perCentCmd;
}


void GateSigmoidalThresholderMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==thresholdCmd )
    GetSigmoidalThresholder()->SetThreshold(thresholdCmd->GetNewDoubleValue(newValue));
  else if ( command==alphaCmd )
    GetSigmoidalThresholder()->SetThresholdAlpha(alphaCmd->GetNewDoubleValue(newValue));
  else if ( command==perCentCmd )
    GetSigmoidalThresholder()->SetThresholdPerCent(perCentCmd->GetNewDoubleValue(newValue));
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
