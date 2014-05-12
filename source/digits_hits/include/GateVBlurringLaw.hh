/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVBlurringLaw_h
#define GateVBlurringLaw_h 1

#include "globals.hh"
#include "GateNamedObject.hh"

/*! \class  GateVBlurringLaw
    \brief  Law giving the resolution in energy


      \sa GateBluring
*/


class GateVBlurringLaw : public GateNamedObject {

	public :
		GateVBlurringLaw(const G4String& itsName);

		virtual ~GateVBlurringLaw() {}
       	virtual G4double ComputeResolution(G4double energy) const = 0;

  		// Implementation of the virtual method in GateNamedObject class
  		void Describe (size_t ident=0);

  		// Pure virtual method called in Describe()
  		virtual void DescribeMyself (size_t ident = 0) const = 0;

};

#endif
