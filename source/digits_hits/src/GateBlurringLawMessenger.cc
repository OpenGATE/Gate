/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBlurringLawMessenger.hh"



GateBlurringLawMessenger::GateBlurringLawMessenger(GateVBlurringLaw* itsBlurringLaw) :
	GateNamedObjectMessenger(itsBlurringLaw)
{
	G4String guidance;

	guidance = G4String("Control for the blurring law '") + GetBlurringLaw()->GetObjectName() + G4String("'");
	GetDirectory()->SetGuidance(guidance.c_str());
}

GateVBlurringLaw* GateBlurringLawMessenger::GetBlurringLaw() const {
	return dynamic_cast<GateVBlurringLaw*>(GetNamedObject());
}

void GateBlurringLawMessenger::SetNewValue(G4UIcommand* cmdName, G4String val) {
	GateNamedObjectMessenger::SetNewValue(cmdName, val);
}
