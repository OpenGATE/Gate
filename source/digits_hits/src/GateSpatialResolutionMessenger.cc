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
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "GateDistributionListManager.hh"
#include "GateVDistribution.hh"



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
	spresolutionXCmd= new G4UIcmdWithADouble(cmdName,this);
	spresolutionXCmd->SetGuidance("Set the resolution ");
	cmdName = GetDirectoryName() + "fwhmY";
	spresolutionYCmd = new G4UIcmdWithADouble(cmdName,this);
	spresolutionYCmd->SetGuidance("Set the resolution in position for gaussian spblurring");

	cmdName = GetDirectoryName() + "fwhmZ";
	spresolutionZCmd = new G4UIcmdWithADouble(cmdName,this);
	spresolutionZCmd->SetGuidance("Set the resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "fwhmXdistrib";
	spresolutionXdistribCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionXdistribCmd->SetGuidance("Set the distribution  resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "fwhmYdistrib";
	spresolutionYdistribCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionYdistribCmd->SetGuidance("Set the  distribution resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "fwhmXdistrib2D";
	spresolutionXdistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionXdistrib2DCmd->SetGuidance("Set the distribution 2D of  spatial resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "fwhmYdistrib2D";
	spresolutionYdistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionYdistrib2DCmd->SetGuidance("Set the  distribution 2D of spatial  resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "confineInsideOfSmallestElement";
	confineCmd = new G4UIcmdWithABool(cmdName,this);
	confineCmd->SetGuidance("To be set true, if you want to moves the outsiders of the crystal after spblurring inside the same crystal");

}


GateSpatialResolutionMessenger::~GateSpatialResolutionMessenger()
{
	delete  spresolutionCmd;
	delete  spresolutionXCmd;
	delete  spresolutionXdistribCmd;
	delete  spresolutionYdistribCmd;
	delete  spresolutionXdistrib2DCmd;
	delete  spresolutionYdistrib2DCmd;
	delete  spresolutionYCmd;
	delete  spresolutionZCmd;
	delete  confineCmd;

}


void GateSpatialResolutionMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	 if ( aCommand==spresolutionCmd )
	    { m_SpatialResolution->SetFWHM(spresolutionCmd->GetNewDoubleValue(newValue)); }
   else if ( aCommand==spresolutionXdistribCmd )
	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
		if (distrib)m_SpatialResolution->SetFWHMxdistrib(distrib);
}
   else if ( aCommand==spresolutionYdistribCmd )
  	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
  		if (distrib)m_SpatialResolution->SetFWHMydistrib(distrib);
  }
   else if ( aCommand==spresolutionXdistrib2DCmd )
	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
		if (distrib)m_SpatialResolution->SetFWHMxdistrib2D(distrib);
}
   else if ( aCommand==spresolutionYdistrib2DCmd )
  	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
  		if (distrib)m_SpatialResolution->SetFWHMydistrib2D(distrib);
  }
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













