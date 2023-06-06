/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateVEffectiveEnergyLaw_h
#define GateVEffectiveEnergyLaw_h 1

#include "globals.hh"
#include "GateNamedObject.hh"
#include "GateDigi.hh"




class GateVEffectiveEnergyLaw : public GateNamedObject {

	public :
         GateVEffectiveEnergyLaw(const G4String& itsName);

        virtual ~ GateVEffectiveEnergyLaw() {}
        virtual G4double ComputeEffectiveEnergy(GateDigi digi) const = 0;

  		// Implementation of the virtual method in GateNamedObject class
  		void Describe (size_t ident=0);

  		// Pure virtual method called in Describe()
  		virtual void DescribeMyself (size_t ident = 0) const = 0;

};

#endif
