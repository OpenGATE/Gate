/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCalibrationMessenger.hh"

#include "GateCalibration.hh"

#include "G4UIcmdWithADouble.hh"

GateCalibrationMessenger::GateCalibrationMessenger(GateCalibration* itsCalibration)
    : GatePulseProcessorMessenger(itsCalibration)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setCalibration";
  calibCmd = new G4UIcmdWithADouble(cmdName,this);
  calibCmd->SetGuidance("Set calibration factor");
}

void GateCalibrationMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==calibCmd )
    { GetCalibration()->SetCalibrationFactor(calibCmd->GetNewDoubleValue(newValue)); }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
