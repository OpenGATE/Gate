/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateDoIBlurrNegExpLawMessenger

  The user can choose the value of ExpInvDecayConst and FWHM needed for the crystal entrance.

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

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
