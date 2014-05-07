/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateInverseSquareBlurringLawMessenger.hh"
#include "GateInverseSquareBlurringLaw.hh"


GateInverseSquareBlurringLawMessenger::GateInverseSquareBlurringLawMessenger(GateInverseSquareBlurringLaw* itsBlurringLaw) :
	GateBlurringLawMessenger(itsBlurringLaw)
{


	G4String cmdName;
	G4String cmdName2;

	cmdName = GetDirectoryName() + "setResolution";
	resolutionCmd = new G4UIcmdWithADouble(cmdName,this);
	resolutionCmd->SetGuidance("Set the resolution in energy for gaussian blurring");

	cmdName2 = GetDirectoryName() + "setEnergyOfReference";
	erefCmd = new G4UIcmdWithADoubleAndUnit(cmdName2,this);
	erefCmd->SetGuidance("Set the energy of reference (in keV) for the selected resolution");
	erefCmd->SetUnitCategory("Energy");
}


GateInverseSquareBlurringLawMessenger::~GateInverseSquareBlurringLawMessenger() {
	delete resolutionCmd;
	delete erefCmd;
}


GateInverseSquareBlurringLaw* GateInverseSquareBlurringLawMessenger::GetInverseSquareBlurringLaw() const {
	return dynamic_cast<GateInverseSquareBlurringLaw*>(GetBlurringLaw());
}



void GateInverseSquareBlurringLawMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==resolutionCmd )
    { GetInverseSquareBlurringLaw()->SetResolution(resolutionCmd->GetNewDoubleValue(newValue)); }
  else if ( command==erefCmd )
    { GetInverseSquareBlurringLaw()->SetEnergyRef(erefCmd->GetNewDoubleValue(newValue)); }
  else
    GateBlurringLawMessenger::SetNewValue(command,newValue);
}
