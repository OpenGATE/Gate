/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateVDoILaw

  This class gives the effective energy for a pulse.

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateVDoILaw.hh"
#include "GateMessageManager.hh"



 GateVDoILaw:: GateVDoILaw(const G4String& itsName) :
	GateNamedObject(itsName)
{

}


void  GateVDoILaw::Describe (size_t ident) {
	GateNamedObject::Describe(ident);

    G4cout << "Law giving the effective energy for a digi.\n";
    DescribeMyself(ident);
}
