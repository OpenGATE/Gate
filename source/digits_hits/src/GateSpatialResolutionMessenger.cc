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
#include "G4UIcmdWithADoubleAndUnit.hh"
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
	spresolutionCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	spresolutionCmd->SetGuidance("Set the resolution (in mm) in position for gaussian spblurring");
	spresolutionCmd->SetUnitCategory("Length");
	cmdName = GetDirectoryName() + "fwhmX";
	spresolutionXCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
	spresolutionXCmd->SetGuidance("Set the resolution (in mm) in position for gaussian spblurring");
	spresolutionXCmd->SetUnitCategory("Length");
	cmdName = GetDirectoryName() + "fwhmY";
	spresolutionYCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	spresolutionYCmd->SetGuidance("Set the resolution in position for gaussian spblurring");
	spresolutionYCmd->SetUnitCategory("Length");
	cmdName = GetDirectoryName() + "fwhmZ";
	spresolutionZCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
	spresolutionZCmd->SetGuidance("Set the resolution in position for gaussian spblurring");
	spresolutionZCmd->SetUnitCategory("Length");
	cmdName = GetDirectoryName() + "fwhmXdistrib";
	spresolutionXdistribCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionXdistribCmd->SetGuidance("Set the distribution  resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "fwhmYdistrib";
	spresolutionYdistribCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionYdistribCmd->SetGuidance("Set the  distribution resolution in position for gaussian spblurring");
	cmdName = GetDirectoryName() + "fwhmXYdistrib2D";
	spresolutionXYdistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionXYdistrib2DCmd->SetGuidance("Set the distribution 2D of  spatial resolution in position for gaussian spblurring");

	cmdName = GetDirectoryName() + "fwhmDistrib2D";
	spresolutionDistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionDistrib2DCmd->SetGuidance("Set the distribution 2D of  spatial resolution in position for gaussian spblurring");


    cmdName = GetDirectoryName()+"nameAxis";
    nameAxisCmd = new G4UIcmdWithAString(cmdName,this);
    nameAxisCmd ->SetGuidance("Provide the name of the axis that will follow the spatial resolution distribution, X, Y, Z, XY, XZ, or YZ");
    nameAxisCmd ->SetCandidates("X Y Z XY XZ YZ");


	cmdName = GetDirectoryName() + "confineInsideOfSmallestElement";
    confineCmd = new G4UIcmdWithABool(cmdName,this);
    confineCmd->SetGuidance("To be set true, if you want to moves the outsiders of the crystal after spblurring inside the same crystal");
}


GateSpatialResolutionMessenger::~GateSpatialResolutionMessenger()
{
	delete  nameAxisCmd;

	delete  spresolutionCmd;
	delete  spresolutionXCmd;
	delete  spresolutionXdistribCmd;
	delete  spresolutionYdistribCmd;
	delete  spresolutionXYdistrib2DCmd;

	delete  spresolutionYCmd;
	delete  spresolutionZCmd;
	delete  confineCmd;

}


void GateSpatialResolutionMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	 if ( aCommand==spresolutionCmd )
	    { m_SpatialResolution->SetFWHM(spresolutionCmd->GetNewDoubleValue(newValue)); }
	 // Handle command for 1D X-distribution resolution
   else if ( aCommand==spresolutionXdistribCmd )
	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
		if (distrib)m_SpatialResolution->SetFWHMxdistrib(distrib);
        }
	 // Handle command for 1D Y-distribution resolution
   else if ( aCommand==spresolutionYdistribCmd )
  	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
  		if (distrib)m_SpatialResolution->SetFWHMydistrib(distrib);
        }
  		// Handle command for 2D XY-distribution resolution
   else if (aCommand == spresolutionXYdistrib2DCmd)
          {GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
        if (distrib) m_SpatialResolution->SetFWHMxydistrib2D(distrib);
        }

   if (aCommand == nameAxisCmd)
	 	      {
	 			m_SpatialResolution->SetNameAxis(newValue);
	 	      }
   else if (aCommand == spresolutionDistrib2DCmd)
             {GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
           if (distrib) m_SpatialResolution->SetFWHMDistrib2D(distrib);
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













