/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



#ifndef GateDepositedEnergyLawMessenger_h
#define GateDepositedEnergyLawMessenger_h

#include "GateEffectiveEnergyLawMessenger.hh"

class GateDepositedEnergyLaw;

class GateDepositedEnergyLawMessenger : public GateEffectiveEnergyLawMessenger {

	public :
        GateDepositedEnergyLawMessenger(GateDepositedEnergyLaw* itsEffectiveEnergyLaw);
        virtual ~GateDepositedEnergyLawMessenger(){};

        GateDepositedEnergyLaw* GetDepositedEnergyLaw() const;

		void SetNewValue(G4UIcommand* aCommand, G4String aString);



};

#endif
