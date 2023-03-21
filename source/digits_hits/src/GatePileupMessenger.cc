/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GatePileupMessenger
    \brief  Messenger for the GatePileup
    \sa GatePileup, GatePileupMessenger
*/


#include "GatePileupMessenger.hh"
#include "GatePileup.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GatePileupMessenger::GatePileupMessenger (GatePileup* Pileup)
:GateClockDependentMessenger(Pileup),
 	 m_Pileup(Pileup)
{
	G4String guidance;
	G4String cmdName;

    cmdName = GetDirectoryName()+"setDepth";
    SetDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetDepthCmd->SetGuidance("Defines the 'depth' of the Pileup");

    cmdName = GetDirectoryName()+"setPileupVolume";
    SetNewVolCmd = new G4UIcmdWithAString(cmdName,this);
    SetNewVolCmd->SetGuidance("Choose a volume (depth) for pileup (e.g. crystal)");

    cmdName = GetDirectoryName()+"setPileup";
    SetPileupCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    SetPileupCmd->SetGuidance("Defines the 'time' of the Pileup");
    SetPileupCmd->SetUnitCategory("Time");

}


GatePileupMessenger::~GatePileupMessenger()
{
	 delete SetNewVolCmd;
	 delete SetDepthCmd;
	 delete SetPileupCmd;
}


void GatePileupMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if( aCommand==SetDepthCmd )
	{ m_Pileup->SetDepth(SetDepthCmd->GetNewIntValue(newValue));}
	else if (aCommand==SetPileupCmd )
	{ m_Pileup->SetPileup(SetPileupCmd->GetNewDoubleValue(newValue));}
	else if (aCommand==SetNewVolCmd)
	{
      m_Pileup->SetVolumeName(newValue);
	}
	else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













