/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GateTimeResolutionMessenger
    \brief  Messenger for the GateTimeResolution

    - GateTimeResolution - by Martin.Rey@epfl.ch (July 2003)

    \sa GateTimeResolution, GateTimeResolutionMessenger
*/

#include "GateTimeResolutionMessenger.hh"
#include "GateTimeResolution.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIdirectory.hh"



GateTimeResolutionMessenger::GateTimeResolutionMessenger (GateTimeResolution* TimeResolution)
:GateClockDependentMessenger(TimeResolution),
 	 m_TimeResolution(TimeResolution)
{
	G4String guidance;
	G4String cmdName;

	cmdName = GetDirectoryName() + "fwhm";
	fwhmCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	fwhmCmd->SetGuidance("Set the temporal resolution with time unity (for expemple: 1 ns) for pulse-discrimination");
	fwhmCmd->SetUnitCategory("Time");

}


GateTimeResolutionMessenger::~GateTimeResolutionMessenger()
{
	  delete fwhmCmd;
}


void GateTimeResolutionMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	 if ( aCommand==fwhmCmd )
	    {
		 m_TimeResolution->SetFWHM(fwhmCmd->GetNewDoubleValue(newValue));
	    }
	  else
		  GateClockDependentMessenger::SetNewValue(aCommand,newValue);
}













