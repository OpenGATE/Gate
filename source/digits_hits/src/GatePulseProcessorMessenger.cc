/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
//GND 2022 Class to Remove

#include "GatePulseProcessorMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateVPulseProcessor.hh"

GatePulseProcessorMessenger::GatePulseProcessorMessenger(GateVPulseProcessor* itsPulseProcessor)
: GateClockDependentMessenger(itsPulseProcessor)
{ 
  G4String guidance;
  G4String cmdName;

  guidance = G4String("Control for the pulse-processor '") + GetPulseProcessor()->GetObjectName() + "'";
  GetDirectory()->SetGuidance(guidance.c_str());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GatePulseProcessorMessenger::~GatePulseProcessorMessenger()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GatePulseProcessorMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  GateClockDependentMessenger::SetNewValue(command,newValue);
}



