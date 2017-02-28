/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTimestampDiscretizerMessenger.hh"

#include "GateTimestampDiscretizer.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateTimestampDiscretizerMessenger::GateTimestampDiscretizerMessenger(GateTimestampDiscretizer* itsTimestampDiscretizer)
    : GatePulseProcessorMessenger(itsTimestampDiscretizer)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setSamplingFrequency";
  frequencyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  frequencyCmd->SetGuidance("Set sampling frequency (in Hz) for timestamp discretization");
  frequencyCmd->SetUnitCategory("Frequency");

  cmdName = GetDirectoryName() + "setSamplingTime";
  timeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  timeCmd->SetGuidance("Set sampling time (in s) for timestamp discretization");
  timeCmd->SetUnitCategory("Time");
}

GateTimestampDiscretizerMessenger::~GateTimestampDiscretizerMessenger()
{
  delete frequencyCmd;
  delete timeCmd;
}

void GateTimestampDiscretizerMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command == frequencyCmd ) {
  	GetTimestampDiscretizer()->SetSamplingFrequency(frequencyCmd->GetNewDoubleValue(newValue));
  } else if ( command == timeCmd ) {
  	GetTimestampDiscretizer()->SetSamplingTime(timeCmd->GetNewDoubleValue(newValue));
  } else {
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
  }
}
