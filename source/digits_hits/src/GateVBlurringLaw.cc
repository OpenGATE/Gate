/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateVBlurringLaw.hh"


/*! \class  GateVBlurringLaw
    \brief  Abstract class for the law giving the energy resolution


    - GateVBlurringLaw - by Henri.Dersarkissian@gmail.com

    - GateVBlurringLaw is the abstract base-class for all energy resolution blurring laws.
      These laws are describing the behaviour of the resolution of the gaussion in energy blurring functions to the energy

    - When developping a new energy blurring law, one should:
      - overload the pure virtual ComputeResolution method;
      - develop a messenger for this blurring law;
      - add the new blurring law to the list of choices available in
      	GateBlurringMessenger.

      \sa GateBlurring, GateLinearBlurringLaw, GateInverseSquareBlurringLaw


*/


GateVBlurringLaw::GateVBlurringLaw(const G4String& itsName) :
	GateNamedObject(itsName)
{

}


void GateVBlurringLaw::Describe (size_t ident) {
	GateNamedObject::Describe(ident);

	G4cout << "Law giving the energy resolution functions of the energy." << G4endl;
	DescribeMyself(ident);
}
