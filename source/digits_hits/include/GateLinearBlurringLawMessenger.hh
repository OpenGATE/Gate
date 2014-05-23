/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateLinearBlurringLawMessenger_h
#define GateLinearBlurringLawMessenger_h

#include "GateBlurringLawMessenger.hh"

class GateLinearBlurringLaw;

class GateLinearBlurringLawMessenger : public GateBlurringLawMessenger {

	public :
		GateLinearBlurringLawMessenger(GateLinearBlurringLaw* itsBlurringLaw);
		virtual ~GateLinearBlurringLawMessenger();

		GateLinearBlurringLaw* GetLinearBlurringLaw() const;

		void SetNewValue(G4UIcommand* aCommand, G4String aString);

	private :
		G4UIcmdWithADouble   *resolutionCmd;
    	G4UIcmdWithADoubleAndUnit   *slopeCmd;
    	G4UIcmdWithADoubleAndUnit   *erefCmd;

};

#endif
