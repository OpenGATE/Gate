/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateThresholderMessenger.hh"

#include "GateThresholder.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateThresholderMessenger::GateThresholderMessenger(GateThresholder* itsThresholder)
    : GatePulseProcessorMessenger(itsThresholder)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setThreshold";
  thresholdCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  thresholdCmd->SetGuidance("Set threshold (in keV) for pulse-discrimination");
  thresholdCmd->SetUnitCategory("Energy");
}


GateThresholderMessenger::~GateThresholderMessenger()
{
  delete thresholdCmd;
}


void GateThresholderMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==thresholdCmd )
    { GetThresholder()->SetThreshold(thresholdCmd->GetNewDoubleValue(newValue)); }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
