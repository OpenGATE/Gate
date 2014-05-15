/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePulseAdderGPUSpectMessenger.hh"

#include "GatePulseAdderGPUSpect.hh"

GatePulseAdderGPUSpectMessenger::GatePulseAdderGPUSpectMessenger(GatePulseAdderGPUSpect *itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{
}


void GatePulseAdderGPUSpectMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
