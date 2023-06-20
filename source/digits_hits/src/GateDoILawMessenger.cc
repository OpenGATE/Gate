/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateDoILawMessenger


  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateDoILawMessenger.hh"



GateDoILawMessenger::GateDoILawMessenger(GateVDoILaw* itsDoILaw) :
    GateNamedObjectMessenger(itsDoILaw)
{
	G4String guidance;

    guidance = G4String("Control for the DoI model '") + GetDoILaw()->GetObjectName() + G4String("'");
	GetDirectory()->SetGuidance(guidance.c_str());
}

GateVDoILaw* GateDoILawMessenger::GetDoILaw() const {
    return dynamic_cast<GateVDoILaw*>(GetNamedObject());
}

void GateDoILawMessenger::SetNewValue(G4UIcommand* cmdName, G4String val) {
	GateNamedObjectMessenger::SetNewValue(cmdName, val);
}
