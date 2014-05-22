/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateInverseSquareBlurringLawMessenger_h
#define GateInverseSquareBlurringLawMessenger_h

#include "GateBlurringLawMessenger.hh"

class GateInverseSquareBlurringLaw;

class GateInverseSquareBlurringLawMessenger : public GateBlurringLawMessenger {

	public :
		GateInverseSquareBlurringLawMessenger(GateInverseSquareBlurringLaw* itsBlurringLaw);
		virtual ~GateInverseSquareBlurringLawMessenger();

		GateInverseSquareBlurringLaw* GetInverseSquareBlurringLaw() const;

		void SetNewValue(G4UIcommand* aCommand, G4String aString);

	private :
		G4UIcmdWithADouble   *resolutionCmd;
    	G4UIcmdWithADoubleAndUnit   *erefCmd;
};

#endif
