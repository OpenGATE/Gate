
#include "GatePulseAdderComptPhotIdealMessenger.hh"

#include "GatePulseAdderComptPhotIdeal.hh"

GatePulseAdderComptPhotIdealMessenger::GatePulseAdderComptPhotIdealMessenger(GatePulseAdderComptPhotIdeal* itsPulseAdder)
    : GatePulseProcessorMessenger(itsPulseAdder)
{
}


void GatePulseAdderComptPhotIdealMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  GatePulseProcessorMessenger::SetNewValue(aCommand,aString);
}
