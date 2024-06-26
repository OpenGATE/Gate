/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDiscretizerModule.cc for more detals
  */



/*!
  \class  GateDiscretizer (by marc.granado@universite-paris-saclay.fr)
  \brief   for discretizing the position within a monolithic crystal

    - For each volume the local position of the interactions within the crystal are  discretized.
	the X,Y,Z vector is translated into the IDs of virtual divisions within the crystal
	occupying the levels of SubmoduleID, CrystalID and LayerID.

*/


#include "GateDiscretizerModuleMessenger.hh"
#include "GateDiscretizerModule.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateDiscretizerModuleMessenger::GateDiscretizerModuleMessenger (GateDiscretizerModule* DiscretizerModule)
:GateClockDependentMessenger(DiscretizerModule),
 	 m_DiscretizerModule(DiscretizerModule)
{
	G4String guidance;
	G4String cmdName;


    cmdName = GetDirectoryName()+"nameAxis";
    nameAxisCmd = new G4UIcmdWithAString(cmdName,this);
    nameAxisCmd ->SetGuidance("Provide the number of axis that need to be discretized, XY or XYZ");
    nameAxisCmd ->SetCandidates("XY XYZ");


    cmdName = GetDirectoryName() + "resolution";
    resCmd = new G4UIcmdWithADouble(cmdName,this);
    resCmd->SetGuidance("Set the size of the resolution for all the selected dimensions");



    cmdName = GetDirectoryName()+"resolutionX";
    resolutionXCmd = new G4UIcmdWithADouble(cmdName,this);
    resolutionXCmd->SetGuidance("Set the size of the resolution for the X dimension");

    cmdName = GetDirectoryName()+"resolutionY";
    resolutionYCmd = new G4UIcmdWithADouble(cmdName,this);
    resolutionYCmd->SetGuidance("Set the size of the resolution for the Y dimension");

    cmdName = GetDirectoryName()+"resolutionZ";
    resolutionZCmd = new G4UIcmdWithADouble(cmdName,this);
    resolutionZCmd->SetGuidance("Set the size of the resolution for the Z dimension");

}


GateDiscretizerModuleMessenger::~GateDiscretizerModuleMessenger()
{
	delete  nameAxisCmd;
	delete  resCmd;
	delete  resolutionXCmd;
	delete  resolutionYCmd;
	delete  resolutionZCmd;
}


void GateDiscretizerModuleMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if (aCommand == nameAxisCmd)
	      {
			m_DiscretizerModule->SetNameAxis(newValue);
	      }

	else if (aCommand == resCmd)
	      {
			m_DiscretizerModule->SetResolution(resCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand == resolutionXCmd)
	      {
			m_DiscretizerModule->SetResolutionX(resCmd->GetNewDoubleValue(newValue));

	      }

	else if (aCommand == resolutionYCmd)
	      {
			m_DiscretizerModule->SetResolutionY(resCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand == resolutionZCmd)
	      {
			m_DiscretizerModule->SetResolutionZ(resCmd->GetNewDoubleValue(newValue));
	      }

	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













