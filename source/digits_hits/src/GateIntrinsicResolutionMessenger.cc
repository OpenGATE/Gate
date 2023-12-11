/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


// OK GND 2022

/*! \class  GateIntrinsicResolutionMessenger
    \brief  Messenger for the GateIntrinsicResolution

    \sa GateEnergyResolution, GateEnergyResolutionMessenger
*/


#include "GateIntrinsicResolutionMessenger.hh"
#include "GateIntrinsicResolution.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIdirectory.hh"



GateIntrinsicResolutionMessenger::GateIntrinsicResolutionMessenger (GateIntrinsicResolution* IntrinsicResolution)
:GateClockDependentMessenger(IntrinsicResolution),
 	 m_IntrinsicResolution(IntrinsicResolution)
{
	G4String guidance;
	G4String cmdName;

    cmdName = GetDirectoryName() + "setIntrinsicResolution";
    resolutionCmd = new G4UIcmdWithADouble(cmdName,this);
    resolutionCmd->SetGuidance("Set the intrinsic resolution in energy for this crystal");

    cmdName = GetDirectoryName() + "setEnergyOfReference";
    erefCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    erefCmd->SetGuidance("Set the energy of reference (in keV) for the selected resolution");
    erefCmd->SetUnitCategory("Energy");

    cmdName = GetDirectoryName() + "setLightOutput";
    lightOutputCmd=new G4UIcmdWithADouble(cmdName,this);
    lightOutputCmd->SetGuidance("Set the Light Output for this crystal (ph/MeV):");

    cmdName = GetDirectoryName() + "setTECoef";
    coeffTECmd=new G4UIcmdWithADouble(cmdName,this);
    coeffTECmd->SetGuidance("Set the coefficient for transfer efficiency");

    cmdName = GetDirectoryName() + "useFileDataForQE";
    newFileQECmd = new G4UIcmdWithAString(cmdName,this);
    newFileQECmd->SetGuidance("Use data from a file to set your quantum efficiency inhomogeneity");

    cmdName = GetDirectoryName() + "setUniqueQE";
    uniqueQECmd = new G4UIcmdWithADouble(cmdName,this);
    uniqueQECmd->SetGuidance("Set an unique quantum efficiency");


    cmdName = GetDirectoryName() + "setGainVariance";
    varianceCmd = new G4UIcmdWithADouble(cmdName,this);
    varianceCmd->SetGuidance("Set an unique quantum efficiency");
    

	cmdName = GetDirectoryName() + "setXtalkEdgesFraction";
	edgesFractionCmd = new G4UIcmdWithADouble(cmdName,this);
	edgesFractionCmd->SetGuidance("Set the fraction of energy which leaves on each edge crystal");

	cmdName = GetDirectoryName() + "setXtalkCornersFraction";
	cornersFractionCmd = new G4UIcmdWithADouble(cmdName,this);
	cornersFractionCmd->SetGuidance("Set the fraction of the energy which leaves on each corner crystal");


}


GateIntrinsicResolutionMessenger::~GateIntrinsicResolutionMessenger()
{
	delete  resolutionCmd;
	delete  erefCmd;
	delete  lightOutputCmd;
	delete  coeffTECmd;
	delete  newFileQECmd;
	delete  uniqueQECmd;
	delete  varianceCmd;
	delete edgesFractionCmd;
	delete cornersFractionCmd;
}


void GateIntrinsicResolutionMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand == resolutionCmd)
	      {
			m_IntrinsicResolution->SetResolution(resolutionCmd->GetNewDoubleValue(newValue));
	      }
	    else if (aCommand ==erefCmd)
	      {
			m_IntrinsicResolution->SetEref(erefCmd->GetNewDoubleValue(newValue));
	      }
	    else if (aCommand ==lightOutputCmd)
	   	      {
	   			m_IntrinsicResolution->SetLightOutput(lightOutputCmd->GetNewDoubleValue(newValue));
	   	      }
	    else if (aCommand ==coeffTECmd)
	   	      {
	   			m_IntrinsicResolution->SetTransferEff(coeffTECmd->GetNewDoubleValue(newValue));
	   	      }
	    else if ( aCommand==newFileQECmd )
	    {
	    	m_IntrinsicResolution->UseFile(newValue);
	    }
	    else if ( aCommand==uniqueQECmd )
	    {
	    	m_IntrinsicResolution->SetUniqueQE(uniqueQECmd->GetNewDoubleValue(newValue));
	    }
	    else if ( aCommand==varianceCmd )
	    {
	    	m_IntrinsicResolution->SetVariance(varianceCmd->GetNewDoubleValue(newValue));
	    }
	    else if (aCommand ==edgesFractionCmd)
	      {
	    	m_IntrinsicResolution->SetEdgesFraction (edgesFractionCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand ==cornersFractionCmd)
	      {
			m_IntrinsicResolution->SetCornersFraction (cornersFractionCmd->GetNewDoubleValue(newValue));
	      }
	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













