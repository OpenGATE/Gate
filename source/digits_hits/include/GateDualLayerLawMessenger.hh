/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com

*/


#ifndef GateDualLayerLawMessenger_h
#define GateDualLayerLawMessenger_h

#include "GateDoILawMessenger.hh"

class GateDualLayerLaw;

class GateDualLayerLawMessenger : public GateDoILawMessenger {

	public :
        GateDualLayerLawMessenger(GateDualLayerLaw* itsDoILaw);
        virtual ~GateDualLayerLawMessenger(){};

        GateDualLayerLaw* GetDualLayerLaw() const;

	void SetNewValue(G4UIcommand* aCommand, G4String aString);



};

#endif
