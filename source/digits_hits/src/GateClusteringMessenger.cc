/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  This is a messenger for Clustering digitizer module
  // OK GND 2022

  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#include "GateClusteringMessenger.hh"
#include "GateClustering.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"


GateClusteringMessenger::GateClusteringMessenger (GateClustering* GateClustering)
:GateClockDependentMessenger(GateClustering),
 m_GateClustering(GateClustering)
{
	G4String guidance;
	G4String cmdName;
	G4String cmdName1;
	G4String cmdName2;

	cmdName = GetDirectoryName() + "setAcceptedDistance";
	pAcceptedDistCmd=new G4UIcmdWithADoubleAndUnit(cmdName,this);
	pAcceptedDistCmd->SetGuidance("Set accepted  distance  for a hit to the center of a cluster to be part of it");
	pAcceptedDistCmd->SetUnitCategory("Length");

	cmdName2 = GetDirectoryName() + "setRejectionMultipleClusters";
	pRejectionMultipleClustersCmd = new  G4UIcmdWithABool(cmdName2,this);
	pRejectionMultipleClustersCmd->SetGuidance("Set to 1 to reject multiple clusters in the same volume");

	cmdName1 = GetDirectoryName()+"positionPolicy";
	ClustCmd = new G4UIcmdWithAString(cmdName1,this);
	ClustCmd->SetGuidance("How to generate position");
	ClustCmd->SetCandidates("energyWeightedCentroid takeEnergyWinner");

}


GateClusteringMessenger::~GateClusteringMessenger()
{
	delete  ClustCmd;
	delete pAcceptedDistCmd;
	delete pRejectionMultipleClustersCmd;
}


void GateClusteringMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand ==ClustCmd)
	{
		m_GateClustering->SetClustering(pAcceptedDistCmd->GetNewDoubleValue(newValue));
	}
	else if ( aCommand==pRejectionMultipleClustersCmd )
	{
		m_GateClustering->SetRejectionFlag(pRejectionMultipleClustersCmd->GetNewBoolValue(newValue));
	}
	else if ( aCommand==pAcceptedDistCmd)
	{
		m_GateClustering->SetAcceptedDistance(pAcceptedDistCmd->GetNewDoubleValue(newValue));
	}
	else
	{
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	}
}













