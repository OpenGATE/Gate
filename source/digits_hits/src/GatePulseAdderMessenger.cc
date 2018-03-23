/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GatePulseAdderMessenger.hh"

#include "GatePulseAdder.hh"

GatePulseAdderMessenger::GatePulseAdderMessenger(GatePulseAdder* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{
}


void GatePulseAdderMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
