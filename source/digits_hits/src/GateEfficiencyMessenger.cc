/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*
   \class  GateEfficiency

 	 ex-EnergyEfficiency and GateLocalEfficiency

  This module apples the efficiency as a function of energy.
  It uses GateVDistribution class to define either analytic
  function or list of values read from a file.


  Added to GND in 2022 by olga.kochebina@cea.fr
  Previous authors are unknown
*/


#include "GateEfficiencyMessenger.hh"
#include "GateEfficiency.hh"

#include "GateDigitizerMgr.hh"

#include "GateVDistribution.hh"
#include "GateDistributionListManager.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIdirectory.hh"



GateEfficiencyMessenger::GateEfficiencyMessenger (GateEfficiency* EnergyEfficiency)
:GateClockDependentMessenger(EnergyEfficiency),
 	 m_EnergyEfficiency(EnergyEfficiency)
{
	G4String guidance;
	G4String cmdName;

	 cmdName = GetDirectoryName()+"setUniqueEfficiency";
	 uniqueEfficiencyCmd = new G4UIcmdWithADouble(cmdName,this);
	 uniqueEfficiencyCmd->SetGuidance("Set unique efficiency");

    cmdName = GetDirectoryName()+"setEfficiency";
    efficiencyCmd = new G4UIcmdWithAString(cmdName,this);
    efficiencyCmd->SetGuidance("Set efficiency from /gate/distribution/");

    cmdName = GetDirectoryName() + "enableLevel";
    enableCmd = new G4UIcmdWithAnInteger(cmdName,this);
    enableCmd->SetGuidance("Set the efficiency");

    cmdName = GetDirectoryName() + "disableLevel";
    disableCmd = new G4UIcmdWithAnInteger(cmdName,this);
    disableCmd->SetGuidance("Set the efficiency");

    cmdName = GetDirectoryName()+"setMode";
    modeCmd = new G4UIcmdWithAString(cmdName,this);
    modeCmd->SetGuidance("How to generate position");
    modeCmd->SetCandidates("energy crystal");


}


GateEfficiencyMessenger::~GateEfficiencyMessenger()
{
	  delete uniqueEfficiencyCmd;
	  delete efficiencyCmd;
	  delete enableCmd;
	  delete disableCmd;
	  delete modeCmd;
}


void GateEfficiencyMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if ( aCommand==uniqueEfficiencyCmd )
	  {
		m_EnergyEfficiency->SetUniqueEfficiency(uniqueEfficiencyCmd->GetNewDoubleValue(newValue));
	  }
	else if ( aCommand==modeCmd)
		  {
			m_EnergyEfficiency->SetMode(newValue);
		  }
	else if (aCommand == efficiencyCmd)
	  {
		GateVDistribution* distrib = (GateVDistribution*)GateDistributionListManager::GetInstance()->FindElementByBaseName(newValue);
		if (distrib) m_EnergyEfficiency->SetEfficiency(distrib);
	  }
	else if ( aCommand==enableCmd )
	  {
		m_EnergyEfficiency->SetLevel(enableCmd->GetNewIntValue(newValue),true);
	  }
	else if ( aCommand==disableCmd)
	  {
		m_EnergyEfficiency->SetLevel(disableCmd->GetNewIntValue(newValue),false);
	  }
	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}









