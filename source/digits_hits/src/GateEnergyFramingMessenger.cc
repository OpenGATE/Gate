/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
	/*! This a messenger for EnergyFraming digitizer module
	 * Previously Thresholder and Upholder
	 *
	 * TODO GND 202
	 * Implement let law option for CC from old GateEnergyThresholder:
	*
	 * if ( law == "solidAngleWeighted" ) {
        return new GateSolidAngleWeightedEnergyLaw(GetEnergyThresholder()->GetObjectName()+ G4String("/solidAngleWeighted"));

    } else if ( law == "depositedEnergy" ) {
        return new GateDepositedEnergyLaw(GetEnergyThresholder()->GetObjectName() + G4String("/depositedEnergy"));
    } else {
	 *
	 *
	 */

#include "GateEnergyFramingMessenger.hh"
#include "GateEnergyFraming.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"



GateEnergyFramingMessenger::GateEnergyFramingMessenger (GateEnergyFraming* EnergyFraming)
:GateClockDependentMessenger(EnergyFraming),
 	 m_EnergyFraming(EnergyFraming)
{
	G4String guidance;
	G4String cmdName;

	//G4cout<< GetDirectoryName()<<G4endl;

	cmdName = GetDirectoryName() + "setMin";
	//G4cout<<cmdName<<G4endl;
	setMinCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	setMinCmd->SetGuidance("Set uphold (in keV) for pulse-limitation");
	setMinCmd->SetUnitCategory("Energy");


	 cmdName = GetDirectoryName() + "setMax";
	 setMaxCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	 setMaxCmd->SetGuidance("Set threshold (in keV) for pulse-discrimination");
	 setMaxCmd->SetUnitCategory("Energy");

	//TODO GND 2022
	//Implement let law option for CC from old GateThresholder
	/*
	cmdName2 = GetDirectoryName() + "setLaw";
	  lawCmd = new G4UIcmdWithAString(cmdName2,this);
	  lawCmd->SetGuidance("Set the law of effective energy  for the threshold");
	 */
}


GateEnergyFramingMessenger::~GateEnergyFramingMessenger()
{
	delete  setMaxCmd;
	delete  setMinCmd;
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
	else
	{
		GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	}
}













