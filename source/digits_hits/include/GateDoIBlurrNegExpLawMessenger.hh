/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/



#ifndef GateDoIBlurrNegExpLawMessenger_h
#define GateDoIBlurrNegExpLawMessenger_h

#include "GateDoILawMessenger.hh"

class GateDoIBlurrNegExpLaw;

class GateDoIBlurrNegExpLawMessenger : public GateDoILawMessenger {

	public :
        GateDoIBlurrNegExpLawMessenger(GateDoIBlurrNegExpLaw* itsDoILaw);
        virtual ~GateDoIBlurrNegExpLawMessenger();

        GateDoIBlurrNegExpLaw* GetDoIBlurrNegExpLaw() const;

	void SetNewValue(G4UIcommand* aCommand, G4String aString);

private:
    G4UIcmdWithADoubleAndUnit   *entFWHMCmd;
    G4UIcmdWithADoubleAndUnit   *expDCmd;



};

#endif
