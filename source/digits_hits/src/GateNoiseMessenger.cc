/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

#include "GateNoiseMessenger.hh"
#include "GateNoise.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

#include "GateVDistribution.hh"
#include "GateDistributionListManager.hh"

GateNoiseMessenger::GateNoiseMessenger (GateNoise* Noise)
:GateClockDependentMessenger(Noise),
 	 m_Noise(Noise)
{
	G4String guidance;
	G4String cmdName;

	 cmdName = GetDirectoryName() + "setDeltaTDistribution";
	 m_deltaTDistribCmd = new G4UIcmdWithAString(cmdName,this);;
	 m_deltaTDistribCmd->SetGuidance("Set the deltaT distribution");

	 cmdName = GetDirectoryName() + "setEnergyDistribution";
	 m_energyDistribCmd = new G4UIcmdWithAString(cmdName,this);;
	 m_energyDistribCmd->SetGuidance("Set the energy distribution");
}


GateNoiseMessenger::~GateNoiseMessenger()
{
	  delete m_deltaTDistribCmd;
	  delete m_energyDistribCmd;
}


void GateNoiseMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if ( aCommand==m_deltaTDistribCmd ){
		GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
		if (distrib) m_Noise->SetDeltaTDistribution(distrib);
	} else if (aCommand==m_energyDistribCmd ){
		GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
		if (distrib) m_Noise->SetEnergyDistribution(distrib);
	}
	else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













