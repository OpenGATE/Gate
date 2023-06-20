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

#ifndef GateVDoILaw_h
#define GateVDoILaw_h 1

#include "globals.hh"
#include "GateNamedObject.hh"
#include "GateDigi.hh"




class GateVDoILaw : public GateNamedObject {

	public :
         GateVDoILaw(const G4String& itsName);

        virtual ~ GateVDoILaw() {}
        virtual void ComputeDoI(GateDigi* digi, G4ThreeVector axis) = 0;

  		// Implementation of the virtual method in GateNamedObject class
  		void Describe (size_t ident=0);

  		// Pure virtual method called in Describe()
  		virtual void DescribeMyself (size_t ident = 0) const = 0;

};

#endif
