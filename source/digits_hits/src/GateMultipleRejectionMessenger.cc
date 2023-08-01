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
	  G4String cmdName2;
	  m_count=0;

	  cmdName = GetDirectoryName() + "setEventRejection";
	  newEventRejCmd = new G4UIcmdWithABool(cmdName,this);
	  newEventRejCmd->SetGuidance("Choose an Event Rejection for  MultipleRejection");

	  cmdName2 = GetDirectoryName() + "setMultipleDefinition";
	  newMultiDefCmd = new G4UIcmdWithAString(cmdName2,this);
	  newMultiDefCmd->SetGuidance("Set multiple definition");
}

GateMultipleRejectionMessenger::~GateMultipleRejectionMessenger()
{
  delete newEventRejCmd;
  for (G4int i=0;i<m_count;i++) {
	 delete MultipleRejectionPolicyCmd[i];
	 delete MultipleDefinitionCmd[i];
   }
}


void GateMultipleRejectionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
	if ( command==newEventRejCmd )
	{
		G4String cmdName2, cmdName3;

		if(m_MultipleRejection) {
			m_volDirectory.push_back(new G4UIdirectory( (GetDirectoryName() + newValue + "/").c_str() ));
			m_volDirectory[m_count]->SetGuidance((G4String("characteristics of ") + newValue).c_str());

			m_name.push_back(newValue);


			cmdName2 = m_volDirectory[m_count]->GetCommandPath() + "setEventRejection";
			MultipleRejectionPolicyCmd.push_back(new G4UIcmdWithABool(cmdName2,this));
			MultipleRejectionPolicyCmd[m_count]->SetGuidance("Set  rejection policy for the chosen volume. When there are multiples the whole event can be rejected or only those interactions in the studied volume ");



			cmdName3 = m_volDirectory[m_count]->GetCommandPath() + "setMultipleDefinition";
			MultipleDefinitionCmd.push_back(new G4UIcmdWithAString(cmdName3,this));
			MultipleDefinitionCmd[m_count]->SetGuidance("Set   the definition of multiples. We can considerer as multiples,  more than one single in the same volume Name or in the same volumeID (for repeaters). ");
			MultipleDefinitionCmd[m_count]->SetCandidates("volumeName volumeID");

			m_count++;
		}
	}
	else
		SetNewValue2(command,newValue);
}

void GateMultipleRejectionMessenger::SetNewValue2(G4UIcommand* command, G4String newValue)
{
  G4int test=0;
  for (G4int i=0;i<m_count;i++)  {
	if ( command==MultipleRejectionPolicyCmd[i] ) {
		m_MultipleRejection->SetRejectionPolicy(m_name[i],  MultipleRejectionPolicyCmd[m_count]->GetNewBoolValue(newValue));
	  test=1;
	}
  }
  if(test==0)
	for (G4int i=0;i<m_count;i++)  {
	  if ( command==MultipleDefinitionCmd[i] ) {
		  m_MultipleRejection->SetMultipleDefinition(m_name[i], newValue);
	test=1;

	  }
	}

  if(test==0)
	  GateClockDependentMessenger::SetNewValue(command,newValue);
}
