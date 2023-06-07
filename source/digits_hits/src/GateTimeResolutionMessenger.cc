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
#include "G4UIcmdWithAString.hh"

#include "G4UIdirectory.hh"



GateTimeResolutionMessenger::GateTimeResolutionMessenger (GateTimeResolution* TimeResolution)
:GateClockDependentMessenger(TimeResolution),
 	 m_TimeResolution(TimeResolution)
{
	G4String guidance;
	G4String cmdName;

	cmdName = GetDirectoryName() + "fwhm";
	fwhmCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	fwhmCmd->SetGuidance("Set the temporal resolution with time unity (for expemple: 1 ns) for digi-discrimination");
	fwhmCmd->SetUnitCategory("Time");

	cmdName = GetDirectoryName() + "CTR";
	ctrCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	ctrCmd->SetGuidance("Set the coincidence time resolution with time unity (for expemple: 1 ns)");
	ctrCmd->SetUnitCategory("Time");

	cmdName = GetDirectoryName() + "DOIdimention4CTR";
	doiCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	doiCmd->SetGuidance("Set dimension that corresponds to your DOI for the CTR calculation.\n"
			"CTR is calculated as the following:\n "
			"CTR=sqrt (2*STR*STR+S*S),\n"
			"where CTR = coincidence time resolution, \n"
			"STR = single time resolution, \n"
			"S = time spread due to geometry dimensions of the detector/DOI in this approximation, i. e. \n"
			"S = speed of light / DOI");
	doiCmd->SetUnitCategory("Length");




}


GateTimeResolutionMessenger::~GateTimeResolutionMessenger()
{
	  delete fwhmCmd;
	  delete ctrCmd;
	  delete doiCmd;
}


void GateTimeResolutionMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	 if ( aCommand==fwhmCmd )
	    {
		 m_TimeResolution->SetFWHM(fwhmCmd->GetNewDoubleValue(newValue));
	    }
	 else if (aCommand==ctrCmd)
	 {
		 m_TimeResolution->SetCTR(ctrCmd->GetNewDoubleValue(newValue));
	 }
	 else if (aCommand==doiCmd)
	 {
		 m_TimeResolution->SetDOI(doiCmd->GetNewDoubleValue(newValue));
	 }

	  else
		  GateClockDependentMessenger::SetNewValue(aCommand,newValue);
}













