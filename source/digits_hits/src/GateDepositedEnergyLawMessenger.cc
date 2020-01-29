/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateDepositedEnergyLawMessenger.hh"
#include "GateDepositedEnergyLaw.hh"


GateDepositedEnergyLawMessenger::GateDepositedEnergyLawMessenger(GateDepositedEnergyLaw* itsEffectiveEnergyLaw) :
    GateEffectiveEnergyLawMessenger(itsEffectiveEnergyLaw)
{


}



GateDepositedEnergyLaw* GateDepositedEnergyLawMessenger::GetDepositedEnergyLaw() const {
    return dynamic_cast<GateDepositedEnergyLaw*>(GetEffectiveEnergyLaw());
}



void GateDepositedEnergyLawMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    GateEffectiveEnergyLawMessenger::SetNewValue(command,newValue);
}
