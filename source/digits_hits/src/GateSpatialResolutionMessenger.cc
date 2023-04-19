/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

#include "GateSpatialResolutionMessenger.hh"
#include "GateSpatialResolution.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIdirectory.hh"



GateSpatialResolutionMessenger::GateSpatialResolutionMessenger (GateSpatialResolution* SpatialResolution)
:GateClockDependentMessenger(SpatialResolution),
 	 m_SpatialResolution(SpatialResolution)
{
	G4String guidance;
	G4String cmdName;

	cmdName = GetDirectoryName() + "fwhm";
	spresolutionCmd = new G4UIcmdWithADouble(cmdName,this);
	spresolutionCmd->SetGuidance("Set the resolution in position for gaussian spblurring");

	cmdName = GetDirectoryName() + "fwhmX";
	spresolutionXCmd = new G4UIcmdWithADouble(cmdName,this);
	spresolutionXCmd->SetGuidance("Set the resolution in position for gaussian spblurring");

	cmdName = GetDirectoryName() + "fwhmY";
	spresolutionYCmd = new G4UIcmdWithADouble(cmdName,this);
	spresolutionYCmd->SetGuidance("Set the resolution in position for gaussian spblurring");

	cmdName = GetDirectoryName() + "fwhmZ";
	spresolutionZCmd = new G4UIcmdWithADouble(cmdName,this);
	spresolutionZCmd->SetGuidance("Set the resolution in position for gaussian spblurring");

	cmdName = GetDirectoryName() + "confineInsideOfSmallestElement";
	confineCmd = new G4UIcmdWithABool(cmdName,this);
	confineCmd->SetGuidance("To be set true, if you want to moves the outsiders of the crystal after spblurring inside the same crystal");

}


GateSpatialResolutionMessenger::~GateSpatialResolutionMessenger()
{
	delete  spresolutionCmd;
	delete  spresolutionXCmd;
	delete  spresolutionYCmd;
	delete  spresolutionZCmd;
	delete  confineCmd;


}


void GateSpatialResolutionMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	 if ( aCommand==spresolutionCmd )
	    { m_SpatialResolution->SetFWHM(spresolutionCmd->GetNewDoubleValue(newValue)); }
	 else if ( aCommand==spresolutionXCmd )
	 	{ m_SpatialResolution->SetFWHMx(spresolutionXCmd->GetNewDoubleValue(newValue)); }
	 else if ( aCommand==spresolutionYCmd )
		{ m_SpatialResolution->SetFWHMy(spresolutionYCmd->GetNewDoubleValue(newValue)); }
	 else if ( aCommand==spresolutionZCmd )
		{ m_SpatialResolution->SetFWHMz(spresolutionZCmd->GetNewDoubleValue(newValue)); }
	  else if ( aCommand==confineCmd )
		{ m_SpatialResolution->ConfineInsideOfSmallestElement(confineCmd->GetNewBoolValue(newValue)); }
	 else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













