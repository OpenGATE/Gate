/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

#include "GateCrosstalkMessenger.hh"
#include "GateCrosstalk.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"


class G4UIcmdWithAString;
class G4UIcmdWithADouble;


GateCrosstalkMessenger::GateCrosstalkMessenger (GateCrosstalk* Crosstalk)
:GateClockDependentMessenger(Crosstalk),
 	 m_Crosstalk(Crosstalk)
{
	G4String guidance;
	G4String cmdName;

	cmdName = GetDirectoryName() + "setEdgesFraction";
	edgesFractionCmd = new G4UIcmdWithADouble(cmdName,this);
	edgesFractionCmd->SetGuidance("Set the fraction of energy which leaves on each edge crystal");

	cmdName = GetDirectoryName() + "setCornersFraction";
	cornersFractionCmd = new G4UIcmdWithADouble(cmdName,this);
	cornersFractionCmd->SetGuidance("Set the fraction of the energy which leaves on each corner crystal");

}


GateCrosstalkMessenger::~GateCrosstalkMessenger()
{
	delete edgesFractionCmd;
	delete cornersFractionCmd;
}


void GateCrosstalkMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand ==edgesFractionCmd)
	      {
			m_Crosstalk->SetEdgesFraction (edgesFractionCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand ==cornersFractionCmd)
	      {
			m_Crosstalk->SetCornersFraction (cornersFractionCmd->GetNewDoubleValue(newValue));
	      }
	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













