/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateSolidAngleWeightedEnergyLawMessenger_h
#define GateSolidAngleWeightedEnergyLawMessenger_h

#include "GateEffectiveEnergyLawMessenger.hh"

class GateSolidAngleWeightedEnergyLaw;

class GateSolidAngleWeightedEnergyLawMessenger : public GateEffectiveEnergyLawMessenger {

	public :
        GateSolidAngleWeightedEnergyLawMessenger(GateSolidAngleWeightedEnergyLaw* itsEffectiveEnergyLaw);
        virtual ~GateSolidAngleWeightedEnergyLawMessenger();

        GateSolidAngleWeightedEnergyLaw* GetSolidAngleWeightedEnergyLaw() const;

		void SetNewValue(G4UIcommand* aCommand, G4String aString);

	private :
        G4UIcmdWithAnInteger   *zSense4ReadoutCmd;
        G4UIcmdWithADoubleAndUnit   *szXCmd;
        G4UIcmdWithADoubleAndUnit   *szYCmd;
};

#endif
