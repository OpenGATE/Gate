/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBlurringMessenger.hh"

#include "GateBlurring.hh"

#include  "G4UIcmdWithADoubleAndUnit.hh"

#include "G4UIcmdWithAString.hh"

#include "GateInverseSquareBlurringLaw.hh"
#include "GateLinearBlurringLaw.hh"

GateBlurringMessenger::GateBlurringMessenger(GateBlurring* itsResolution)
    : GatePulseProcessorMessenger(itsResolution)
{

  G4String cmdName;


  cmdName = GetDirectoryName() + "setLaw";
  lawCmd = new G4UIcmdWithAString(cmdName,this);
  lawCmd->SetGuidance("Set the law of energy resolution for gaussian blurring");

}


GateBlurringMessenger::~GateBlurringMessenger()
{
  delete lawCmd;
}


GateVBlurringLaw* GateBlurringMessenger::CreateBlurringLaw(const G4String& law) {

	if ( law == "inverseSquare" ) {
		return new GateInverseSquareBlurringLaw(GetBlurring()->GetObjectName() + G4String("/inverseSquare"));
	} else if ( law == "linear" ) {
		return new GateLinearBlurringLaw(GetBlurring()->GetObjectName() + G4String("/linear"));
	} else {
		G4cerr << "No match for '" << law << "' blurring law." << G4endl;
		G4cerr << "Candidates are: inverseSquare linear" << G4endl;
	}

	return NULL;
}

void GateBlurringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==lawCmd )
    {
    	GateVBlurringLaw* a_blurringLaw = CreateBlurringLaw(newValue);
   		if (a_blurringLaw != NULL) {
    		GetBlurring()->SetBlurringLaw(a_blurringLaw);
  	  	}
  	}
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
