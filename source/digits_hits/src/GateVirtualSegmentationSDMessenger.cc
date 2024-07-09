/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateVirtualSegmentationSD.cc for more detals
  */



/*!
  \class  GateVirtualSegmentationSD (by marc.granado@universite-paris-saclay.fr)
  \brief   for discretizing the position through a virtual segmentation of a monolithic crystal

    - For each volume the local position of the interactions within the crystal are virtually segmented.
	the X,Y,Z vector is translated into the IDs of virtual divisions within the crystal
	occupying the levels of SubmoduleID, CrystalID and LayerID.
*/


#include "GateVirtualSegmentationSDMessenger.hh"
#include "GateVirtualSegmentationSD.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"



GateVirtualSegmentationSDMessenger::GateVirtualSegmentationSDMessenger (GateVirtualSegmentationSD* VirtualSegmentationSD)
:GateClockDependentMessenger(VirtualSegmentationSD),
 	 m_VirtualSegmentationSD(VirtualSegmentationSD)
{
	G4String guidance;
	G4String cmdName;


    cmdName = GetDirectoryName()+"nameAxis";
    nameAxisCmd = new G4UIcmdWithAString(cmdName,this);
    nameAxisCmd ->SetGuidance("Provide the number of axis that need to be virtually segmented, XYZ, XY, XZ, or YZ");
    nameAxisCmd ->SetCandidates("XYZ XY XZ YZ");

    cmdName = GetDirectoryName() + "useMacroGenerator";
    useMacroGeneratorCmd = new G4UIcmdWithABool(cmdName,this);
    useMacroGeneratorCmd->SetGuidance("To be set true, if there is a need to create a new geometry macro for CASToR");


    cmdName = GetDirectoryName() + "pitch";
    pitchCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);//new G4UIcmdWithADouble(cmdName,this);
    pitchCmd->SetGuidance("Set the size of the pitch for all the selected dimensions");



    cmdName = GetDirectoryName()+"pitchX";
    pitchXCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);//new G4UIcmdWithADouble(cmdName,this);
    pitchXCmd->SetGuidance("Set the size of the pitch for the X dimension");

    cmdName = GetDirectoryName()+"pitchY";
    pitchYCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);//new G4UIcmdWithADouble(cmdName,this);
    pitchYCmd->SetGuidance("Set the size of the pitch for the Y dimension");

    cmdName = GetDirectoryName()+"pitchZ";
    pitchZCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);//new G4UIcmdWithADouble(cmdName,this);
    pitchZCmd->SetGuidance("Set the size of the pitch for the Z dimension");

}


GateVirtualSegmentationSDMessenger::~GateVirtualSegmentationSDMessenger()
{
	delete  nameAxisCmd;
	delete 	useMacroGeneratorCmd;
	delete  pitchCmd;
	delete  pitchXCmd;
	delete  pitchYCmd;
	delete  pitchZCmd;
}


void GateVirtualSegmentationSDMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if (aCommand == nameAxisCmd)
	      {
			m_VirtualSegmentationSD->SetNameAxis(newValue);
	      }

	else if (aCommand == pitchCmd)
	      {
			m_VirtualSegmentationSD->SetPitch(pitchCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand == pitchXCmd)
	      {
			m_VirtualSegmentationSD->SetPitchX(pitchCmd->GetNewDoubleValue(newValue));

	      }

	else if (aCommand == pitchYCmd)
	      {
			m_VirtualSegmentationSD->SetPitchY(pitchCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand == pitchZCmd)
	      {
			m_VirtualSegmentationSD->SetPitchZ(pitchCmd->GetNewDoubleValue(newValue));
	      }
	else if ( aCommand==useMacroGeneratorCmd )
			{ m_VirtualSegmentationSD->SetUseMacroGenerator(useMacroGeneratorCmd->GetNewBoolValue(newValue)); }
	    else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













