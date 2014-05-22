/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDiscretizerMessenger.hh"

#include "GateDiscretizer.hh"
#include "G4UIcmdWithAnInteger.hh"

GateDiscretizerMessenger::GateDiscretizerMessenger(GateDiscretizer* itsDiscretizer)
    : GatePulseProcessorMessenger(itsDiscretizer)
{
}


GateDiscretizerMessenger::~GateDiscretizerMessenger()
{
}

void GateDiscretizerMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
    GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
