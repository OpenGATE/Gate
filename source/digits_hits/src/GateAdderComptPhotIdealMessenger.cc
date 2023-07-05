/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  This is a messenger for AdderComptPhotIdeal digitizer module
  // OK GND 2022

  Last modification (Adaptation to GND): July 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateAdderComptPhotIdealMessenger.hh"
#include "GateAdderComptPhotIdeal.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"



GateAdderComptPhotIdealMessenger::GateAdderComptPhotIdealMessenger (GateAdderComptPhotIdeal* GateAdderComptPhotIdeal)
:GateClockDependentMessenger(GateAdderComptPhotIdeal),
 	 m_GateAdderComptPhotIdeal(GateAdderComptPhotIdeal)
{
    G4String cmdName;

    cmdName = GetDirectoryName()+"rejectEvtOtherProcesses";
    pRejectionPolicyCmd=new  G4UIcmdWithABool(cmdName,this);
    pRejectionPolicyCmd->SetGuidance("Set to 1 to reject events with at least one primary interaction different from C or P");

}


GateAdderComptPhotIdealMessenger::~GateAdderComptPhotIdealMessenger()
{
	delete  pRejectionPolicyCmd;
}


void GateAdderComptPhotIdealMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand ==pRejectionPolicyCmd)
	{
		m_GateAdderComptPhotIdeal->SetEvtRejectionPolicy(pRejectionPolicyCmd->GetNewBoolValue(newValue));
	}
	else
	{
		GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	}
}






