/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022


#include "GateDigitizerMergerMessenger.hh"
#include "GateDigitizerMerger.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateDigitizerMergerMessenger::GateDigitizerMergerMessenger (GateDigitizerMerger* DigitizerMerger)
:GateClockDependentMessenger(DigitizerMerger),
 	 m_DigitizerMerger(DigitizerMerger)
{
	G4String guidance;
	G4String cmdName;

	cmdName = GetDirectoryName()+"addInput";
    addCollCmd = new G4UIcmdWithAString(cmdName,this);
    addCollCmd->SetGuidance("Select input collection");

}


GateDigitizerMergerMessenger::~GateDigitizerMergerMessenger()
{
	delete  addCollCmd;
}


void GateDigitizerMergerMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if(aCommand ==addCollCmd)
	      {
			m_DigitizerMerger->AddInputCollection(newValue);
	      }
	    else
	    {

	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













