/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDummyDigitizerModule.cc for more detals
  */

#include "GateDummyDigitizerModuleMessenger.hh"
#include "GateDummyDigitizerModule.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateDummyDigitizerModuleMessenger::GateDummyDigitizerModuleMessenger (GateDummyDigitizerModule* DummyDigitizerModule)
:GateClockDependentMessenger(DummyDigitizerModule),
 	 m_DummyDigitizerModule(DummyDigitizerModule)
{
	G4String guidance;
	G4String cmdName;

    cmdName = GetDirectoryName()+"positionPolicy";
    dummyCmd = new G4UIcmdWithAString(cmdName,this);
    dummyCmd->SetGuidance("How to generate position");
    dummyCmd->SetCandidates("energyWeightedCentroid takeEnergyWinner");

}


GateDummyDigitizerModuleMessenger::~GateDummyDigitizerModuleMessenger()
{
	delete  dummyCmd;
}


void GateDummyDigitizerModuleMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand ==dummyCmd)
	      {
			m_DummyDigitizerModule->SetDummyParameter(newValue);
	      }
	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













