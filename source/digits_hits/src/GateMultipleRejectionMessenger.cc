/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  This is a messenger for MultipleRejection digitizer module

  Last modification (Adaptation to GND): August 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateMultipleRejectionMessenger.hh"
#include "GateMultipleRejection.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

#include "G4UIcmdWithABool.hh"



GateMultipleRejectionMessenger::GateMultipleRejectionMessenger (GateMultipleRejection* MultipleRejection)
:GateClockDependentMessenger(MultipleRejection),
 	 m_MultipleRejection(MultipleRejection)
{
	  G4String guidance;
	  G4String cmdName;

	  cmdName = GetDirectoryName() + "setEventRejection";
	  newEventRejCmd = new G4UIcmdWithABool(cmdName,this);
	  newEventRejCmd->SetGuidance("Choose an Event Rejection for  MultipleRejection");

	  cmdName = GetDirectoryName() + "setMultipleDefinition";
	  newMultiDefCmd = new G4UIcmdWithAString(cmdName,this);
	  newMultiDefCmd->SetGuidance("Set multiple definition");
}

GateMultipleRejectionMessenger::~GateMultipleRejectionMessenger()
{
  delete newEventRejCmd;
  delete newMultiDefCmd;

}


void GateMultipleRejectionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
	if (command ==newEventRejCmd)
	      {
			m_MultipleRejection->SetMultipleRejection(G4UIcmdWithABool::GetNewBoolValue(newValue));
	      }
	else if (command ==newMultiDefCmd)
	      {
			m_MultipleRejection->SetMultipleDefinition(newValue);
	      }
	else {
	    	GateClockDependentMessenger::SetNewValue(command,newValue);
	    }
}
