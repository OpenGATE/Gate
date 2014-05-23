/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceMultiplesKillerMessenger.hh"

#include "GateCoincidenceMultiplesKiller.hh"


GateCoincidenceMultiplesKillerMessenger::GateCoincidenceMultiplesKillerMessenger(GateCoincidenceMultiplesKiller* itsMultiplesKiller)
    : GateClockDependentMessenger(itsMultiplesKiller)
{
}


GateCoincidenceMultiplesKillerMessenger::~GateCoincidenceMultiplesKillerMessenger()
{
}


void GateCoincidenceMultiplesKillerMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
