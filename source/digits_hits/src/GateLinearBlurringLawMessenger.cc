/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLinearBlurringLawMessenger.hh"
#include "GateLinearBlurringLaw.hh"


GateLinearBlurringLawMessenger::GateLinearBlurringLawMessenger(GateLinearBlurringLaw* itsBlurringLaw) :
	GateBlurringLawMessenger(itsBlurringLaw)
{
	G4String cmdName;
	G4String cmdName2;
	G4String cmdName3;

	cmdName = GetDirectoryName() + "setResolution";
	resolutionCmd = new G4UIcmdWithADouble(cmdName,this);
	resolutionCmd->SetGuidance("Set the resolution in energy for gaussian blurring");

	cmdName2 = GetDirectoryName() + "setEnergyOfReference";
	erefCmd = new G4UIcmdWithADoubleAndUnit(cmdName2,this);
	erefCmd->SetGuidance("Set the energy of reference (with unit) for the selected resolution");
	erefCmd->SetUnitCategory("Energy");

	cmdName3 = GetDirectoryName() + "setSlope";
	slopeCmd = new G4UIcmdWithADoubleAndUnit(cmdName3,this);
	slopeCmd->SetGuidance("Set the slope of the linear law (with unit) for gaussian blurring");
}


GateLinearBlurringLawMessenger::~GateLinearBlurringLawMessenger() {
	delete resolutionCmd;
	delete erefCmd;
	delete slopeCmd;
}


GateLinearBlurringLaw* GateLinearBlurringLawMessenger::GetLinearBlurringLaw() const {
	return dynamic_cast<GateLinearBlurringLaw*>(GetBlurringLaw());
}



void GateLinearBlurringLawMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==resolutionCmd )
    { GetLinearBlurringLaw()->SetResolution(resolutionCmd->GetNewDoubleValue(newValue)); }
  else if ( command==erefCmd )
    { GetLinearBlurringLaw()->SetEnergyRef(erefCmd->GetNewDoubleValue(newValue)); }
  else if ( command==slopeCmd )
  	{ GetLinearBlurringLaw()->SetSlope(slopeCmd->GetNewDoubleValue(newValue)); }
  else
    GateBlurringLawMessenger::SetNewValue(command,newValue);
}
