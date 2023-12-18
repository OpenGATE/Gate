/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateTimeDelayMessenger.hh"
#include "GateTimeDelay.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateTimeDelayMessenger::GateTimeDelayMessenger (GateTimeDelay* TimeDelay)
:GateClockDependentMessenger(TimeDelay),
 	 m_TimeDelay(TimeDelay)
{
	G4String guidance;
	G4String cmdName;

    cmdName = GetDirectoryName()+"setTimeDelay";
    TimeDCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    TimeDCmd->SetGuidance("Set the time delay");

}


GateTimeDelayMessenger::~GateTimeDelayMessenger()
{
	delete  TimeDCmd;
}


void GateTimeDelayMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand == TimeDCmd)
	      {
			m_TimeDelay->SetTimeDelay(TimeDCmd->GetNewDoubleValue(newValue));
	      }
	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}



