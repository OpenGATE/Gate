/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateEffectiveEnergyLawMessenger.hh"



GateEffectiveEnergyLawMessenger::GateEffectiveEnergyLawMessenger(GateVEffectiveEnergyLaw* itsEffectiveEnergyLaw) :
    GateNamedObjectMessenger(itsEffectiveEnergyLaw)
{
	G4String guidance;

    guidance = G4String("Control for the effective energy law '") + GetEffectiveEnergyLaw()->GetObjectName() + G4String("'");
	GetDirectory()->SetGuidance(guidance.c_str());
}

GateVEffectiveEnergyLaw* GateEffectiveEnergyLawMessenger::GetEffectiveEnergyLaw() const {
    return dynamic_cast<GateVEffectiveEnergyLaw*>(GetNamedObject());
}

void GateEffectiveEnergyLawMessenger::SetNewValue(G4UIcommand* cmdName, G4String val) {
	GateNamedObjectMessenger::SetNewValue(cmdName, val);
}
