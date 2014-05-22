/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTemporalResolutionMessenger.hh"

#include "GateTemporalResolution.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateTemporalResolutionMessenger::GateTemporalResolutionMessenger(GateTemporalResolution* itsTemporalResolution)
    : GatePulseProcessorMessenger(itsTemporalResolution)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setTimeResolution";
  timeResolutionCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  timeResolutionCmd->SetGuidance("Set the temporal resolution with time unity (for expemple: 1 ns) for pulse-discrimination");
  timeResolutionCmd->SetUnitCategory("Time");
}


GateTemporalResolutionMessenger::~GateTemporalResolutionMessenger()
{
  delete timeResolutionCmd;
}


void GateTemporalResolutionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==timeResolutionCmd )
    { GetTemporalResolution()->SetTimeResolution(timeResolutionCmd->GetNewDoubleValue(newValue)); }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
