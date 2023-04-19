/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GatePulseAdderMessenger
    \brief  Messenger for the GatePulseAdder

    - GatePulseAdderMessenger - by Daniel.Strul@iphe.unil.ch

    \sa GatePulseAdder, GatePulseProcessorMessenger
*/

#include "GateAdderMessenger.hh"
#include "GateAdder.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateAdderMessenger::GateAdderMessenger (GateAdder* adder)
:GateClockDependentMessenger(adder), m_Adder(adder)
{
	G4String guidance;
	G4String cmdName;


    cmdName = GetDirectoryName()+"positionPolicy";
    positionPolicyCmd = new G4UIcmdWithAString(cmdName,this);
    positionPolicyCmd->SetGuidance("How to generate position");
    positionPolicyCmd->SetCandidates("energyWeightedCentroid takeEnergyWinner");

}


GateAdderMessenger::~GateAdderMessenger()
{
	delete positionPolicyCmd;
}


void GateAdderMessenger::SetNewValue(G4UIcommand * aCommand,G4String aString)
{
	if (aCommand ==positionPolicyCmd)
	      {
			m_Adder->SetPositionPolicy(aString);
	      }
	else
	{
		GateClockDependentMessenger::SetNewValue(aCommand,aString);
	}
}













