/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateVBlurringLawMessenger_h
#define GateVBlurringLawMessenger_h

#include "GateNamedObjectMessenger.hh"
#include "GateVBlurringLaw.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"



class GateBlurringLawMessenger : public GateNamedObjectMessenger {

	public :
		GateBlurringLawMessenger(GateVBlurringLaw* itsBlurringLaw);
		virtual ~GateBlurringLawMessenger() {}

		GateVBlurringLaw* GetBlurringLaw() const;
		void SetNewValue(G4UIcommand* cmdName, G4String val);


	private :


};

#endif
