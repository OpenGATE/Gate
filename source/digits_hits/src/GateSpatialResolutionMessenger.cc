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
	cmdName = GetDirectoryName() + "fwhmZdistrib";
	spresolutionZdistribCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionZdistribCmd->SetGuidance("Set the  distribution resolution in position for gaussian spblurring");

	
	cmdName = GetDirectoryName() + "fwhmXDistrib2D";
	spresolutionXDistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionXDistrib2DCmd->SetGuidance("Set the 2D distribution for X axis (expects a 2D distribution object)");
	cmdName = GetDirectoryName() + "fwhmYDistrib2D";
	spresolutionYDistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionYDistrib2DCmd->SetGuidance("Set the 2D distribution for Y axis (expects a 2D distribution object)");
	cmdName = GetDirectoryName() + "fwhmZDistrib2D";
	spresolutionZDistrib2DCmd = new G4UIcmdWithAString(cmdName,this);
	spresolutionZDistrib2DCmd->SetGuidance("Set the 2D distribution for Z axis (expects a 2D distribution object)");


	cmdName = GetDirectoryName()+"nameAxis";
	nameAxisCmd = new G4UIcmdWithAString(cmdName,this);
	nameAxisCmd ->SetGuidance("Provide the coordinate pair used by 2D distributions: 'XZ' or 'YZ'. Default is 'YZ'.");
	nameAxisCmd ->SetCandidates("XZ YZ");


	cmdName = GetDirectoryName() + "confineInsideOfSmallestElement";
    confineCmd = new G4UIcmdWithABool(cmdName,this);
    confineCmd->SetGuidance("To be set true, if you want to moves the outsiders of the crystal after spblurring inside the same crystal");

    cmdName = GetDirectoryName() + "useTruncatedGaussian";
    useTruncatedGaussianCmd = new G4UIcmdWithABool(cmdName,this);
    useTruncatedGaussianCmd->SetGuidance("To be set true, if you want to use a truncated Gaussian distribution to keep the blurring within the same crystal");
}




GateSpatialResolutionMessenger::~GateSpatialResolutionMessenger()
{

	delete  spresolutionCmd;

	delete  spresolutionXCmd;
	delete  spresolutionYCmd;
	delete  spresolutionZCmd;

	delete  spresolutionXdistribCmd;
	delete  spresolutionYdistribCmd;
	delete  spresolutionZdistribCmd;

	delete  spresolutionXDistrib2DCmd;
	delete  spresolutionYDistrib2DCmd;
	delete  spresolutionZDistrib2DCmd;

	delete  nameAxisCmd;
	

	delete  confineCmd;
	delete	useTruncatedGaussianCmd;

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
   else if ( aCommand==spresolutionZdistribCmd )
  	 	{ GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
  		if (distrib)m_SpatialResolution->SetFWHMzdistrib(distrib);
        }
  		// Handle command for 2D-distribution resolution

   if (aCommand == nameAxisCmd)
 	     {
 		// Only accept the PET-relevant options
 		if (newValue == "XZ" || newValue == "YZ") {
 			m_SpatialResolution->SetNameAxis(newValue);
 		} else {
 			G4cout << "***ERROR*** GateSpatialResolution::SetNewValue: nameAxis must be 'XZ' or 'YZ'\n"<< G4endl;
 		}
 	     }
	 else if (aCommand == spresolutionXDistrib2DCmd)
						 {
							 GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
							 if (distrib) m_SpatialResolution->SetFWHMXDistrib2D(distrib);
						 }
	 else if (aCommand == spresolutionYDistrib2DCmd)
						 {
							 GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
							 if (distrib) m_SpatialResolution->SetFWHMYDistrib2D(distrib);
						 }
	 else if (aCommand == spresolutionZDistrib2DCmd)
						 {
							 GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
							 if (distrib) m_SpatialResolution->SetFWHMZDistrib2D(distrib);
						 }

   else if ( aCommand==spresolutionXCmd )
   		{ m_SpatialResolution->SetFWHMx(spresolutionXCmd->GetNewDoubleValue(newValue)); }
	 else if ( aCommand==spresolutionYCmd )
		{ m_SpatialResolution->SetFWHMy(spresolutionYCmd->GetNewDoubleValue(newValue)); }
	 else if ( aCommand==spresolutionZCmd )
		{ m_SpatialResolution->SetFWHMz(spresolutionZCmd->GetNewDoubleValue(newValue)); }
	  else if ( aCommand==confineCmd )
		{ m_SpatialResolution->ConfineInsideOfSmallestElement(confineCmd->GetNewBoolValue(newValue)); }
	  else if ( aCommand==useTruncatedGaussianCmd )
	  		{ m_SpatialResolution->SetUseTruncatedGaussian(useTruncatedGaussianCmd->GetNewBoolValue(newValue)); }
	 else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













