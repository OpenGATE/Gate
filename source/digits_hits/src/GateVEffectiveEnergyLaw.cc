/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateVEffectiveEnergyLaw.hh"
#include "GateMessageManager.hh"



 GateVEffectiveEnergyLaw:: GateVEffectiveEnergyLaw(const G4String& itsName) :
	GateNamedObject(itsName)
{

}


void  GateVEffectiveEnergyLaw::Describe (size_t ident) {
	GateNamedObject::Describe(ident);

    G4cout << "Law giving the effective energy  for a pulse.\n";
	DescribeMyself(ident);
}
