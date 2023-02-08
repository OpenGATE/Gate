/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
// GND 2022 Class to remove
#include "GateHitConvertorMessenger.hh"

#include "G4UIdirectory.hh"

#include "GateHitConvertor.hh"

GateHitConvertorMessenger::GateHitConvertorMessenger(GateHitConvertor* itsHitConvertor)
    : GateClockDependentMessenger(itsHitConvertor)
{
  G4String guidance = G4String("Controls the units that converts hits into pulses.");
  GetDirectory()->SetGuidance(guidance.c_str());

}


void GateHitConvertorMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
  GateClockDependentMessenger::SetNewValue(aCommand,aString);
}
