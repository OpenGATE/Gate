/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateOpticalAdderMessenger.hh"
#include "GateOpticalAdder.hh"

GateOpticalAdderMessenger::GateOpticalAdderMessenger(GateOpticalAdder* itsPulseAdder) :
  GatePulseProcessorMessenger(itsPulseAdder)
{}

void GateOpticalAdderMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{ GatePulseProcessorMessenger::SetNewValue(aCommand,aString);}

#endif
