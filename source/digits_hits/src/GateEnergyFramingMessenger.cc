/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  This is a messenger for EnergyFraming digitizer module
  Previously Thresholder and Upholder
  // OK GND 2022

  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#include "GateEnergyFramingMessenger.hh"
#include "GateEnergyFraming.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "GateSolidAngleWeightedEnergyLaw.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"


GateEnergyFramingMessenger::GateEnergyFramingMessenger (GateEnergyFraming* EnergyFraming)
:GateClockDependentMessenger(EnergyFraming),
 	 m_EnergyFraming(EnergyFraming)
{
	G4String guidance;
	G4String cmdName;
	G4String cmdName2;


	cmdName = GetDirectoryName() + "setMin";
	setMinCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	setMinCmd->SetGuidance("Set uphold (in keV) for pulse-limitation");
	setMinCmd->SetUnitCategory("Energy");


	 cmdName = GetDirectoryName() + "setMax";
	 setMaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	 setMaxCmd->SetGuidance("Set threshold (in keV) for pulse-discrimination");
	 setMaxCmd->SetUnitCategory("Energy");


	 cmdName2 = GetDirectoryName() + "setLaw";
	 setLawCmd = new G4UIcmdWithAString(cmdName2,this);
	 setLawCmd->SetGuidance("Set the law of effective energy  for the framing");

}


GateEnergyFramingMessenger::~GateEnergyFramingMessenger()
{
	delete  setMaxCmd;
	delete  setMinCmd;
	delete	setLawCmd;
}


GateVEffectiveEnergyLaw* GateEnergyFramingMessenger::SetEnergyFLaw(const G4String& law)
{
	if ( law == "solidAngleWeighted" )
	{
	     return new GateSolidAngleWeightedEnergyLaw(m_EnergyFraming->GetObjectName()+ G4String("/solidAngleWeighted"));
	}
	else if ( law == "depositedEnergy" )
	{
	    ;
	}
	return NULL;
}

void GateEnergyFramingMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand ==setMinCmd)
	{
		m_EnergyFraming->SetMin(setMinCmd->GetNewDoubleValue(newValue));
	}
	else if (aCommand ==setMaxCmd)
	{
		m_EnergyFraming->SetMax(setMaxCmd->GetNewDoubleValue(newValue));
	}
	else if (aCommand ==setLawCmd)
		{
			GateVEffectiveEnergyLaw* a_Law = SetEnergyFLaw(newValue);
			if (a_Law != NULL)
			{
				m_EnergyFraming->SetEnergyFLaw(a_Law);
			}
		}
	else
	{
		GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	}
}













