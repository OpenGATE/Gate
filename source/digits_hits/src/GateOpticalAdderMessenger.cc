/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateOpticalAdder.hh"
#include "GateOpticalAdderMessenger.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateOpticalAdderMessenger::GateOpticalAdderMessenger (GateOpticalAdder* OpticalAdder)
:GateClockDependentMessenger(OpticalAdder),
 	 m_OpticalAdder(OpticalAdder)
{

}


GateOpticalAdderMessenger::~GateOpticalAdderMessenger()
{
}


void GateOpticalAdderMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);

}
#endif











