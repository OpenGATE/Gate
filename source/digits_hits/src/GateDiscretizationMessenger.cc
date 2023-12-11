/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateDiscretization.cc for more detals
  
  // OK GND 2022
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateDiscretizationMessenger.hh"
#include "GateDiscretization.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"



GateDiscretizationMessenger::GateDiscretizationMessenger (GateDiscretization* GateDiscretization)
:GateClockDependentMessenger(GateDiscretization),
 	 m_GateDiscretization(GateDiscretization)
{

	G4String guidance;
	G4String cmdName3;
	G4String cmdName4;
	G4String cmdName5;
	G4String cmdName6;
	G4String cmdName7;
	G4String cmdName8;
	G4String cmdName9;
	G4String cmdName10;

    	cmdName3 = GetDirectoryName() + "setNumberStripsX";
	pNumberStripsX = new G4UIcmdWithAnInteger(cmdName3,this);
	pNumberStripsX->SetGuidance("Set Number of Strips in X direction");


	cmdName4 = GetDirectoryName() + "setNumberStripsY";
	pNumberStripsY = new G4UIcmdWithAnInteger(cmdName4,this);
	pNumberStripsY->SetGuidance("Set Number of Strips in Y direction");


	cmdName4 = GetDirectoryName() + "setNumberStripsZ";
	pNumberStripsZ = new G4UIcmdWithAnInteger(cmdName4,this);
	pNumberStripsZ->SetGuidance("Set Number of Strips in Z direction");


    	cmdName5 = GetDirectoryName() + "setStripOffsetX";
	pStripOffsetX =  new G4UIcmdWithADoubleAndUnit(cmdName5,this);
	pStripOffsetX->SetGuidance("Set offset of the strip in X direction from negative axis");
	pStripOffsetX->SetUnitCategory("Length");

	cmdName6 = GetDirectoryName() + "setStripOffsetY";
	pStripOffsetY =  new G4UIcmdWithADoubleAndUnit(cmdName6,this);
	pStripOffsetY->SetGuidance("Set offset of the strip in Y direction from negative axis");
	pStripOffsetY->SetUnitCategory("Length");


	cmdName6 = GetDirectoryName() + "setStripOffsetZ";
	pStripOffsetZ =  new G4UIcmdWithADoubleAndUnit(cmdName6,this);
	pStripOffsetZ->SetGuidance("Set offset of the strip in Y direction from negative axis");
	pStripOffsetZ->SetUnitCategory("Length");


	cmdName7 = GetDirectoryName() + "setStripWidthX";
	pStripWidthX =  new G4UIcmdWithADoubleAndUnit(cmdName7,this);
	pStripWidthX->SetGuidance("Set width of the strip in X direction");
	pStripWidthX->SetUnitCategory("Length");

	cmdName8 = GetDirectoryName() + "setStripWidthY";
	pStripWidthY =  new G4UIcmdWithADoubleAndUnit(cmdName8,this);
	pStripWidthY->SetGuidance("Set width of the strip in Y direction");
	pStripWidthY->SetUnitCategory("Length");

	cmdName8 = GetDirectoryName() + "setStripWidthZ";
	pStripWidthZ =  new G4UIcmdWithADoubleAndUnit(cmdName8,this);
	pStripWidthZ->SetGuidance("Set width of the strip in Z direction");
	pStripWidthZ->SetUnitCategory("Length");

	cmdName9 = GetDirectoryName() + "setNumberReadOutBlocksX";
	pNumberReadOutBlocksX =  new G4UIcmdWithAnInteger(cmdName9,this);
	pNumberReadOutBlocksX->SetGuidance("Set Number of ReadOut blocks in X direction");


	cmdName10 = GetDirectoryName() + "setNumberReadOutBlocksY";
	pNumberReadOutBlocksY =  new G4UIcmdWithAnInteger(cmdName10,this);
	pNumberReadOutBlocksY->SetGuidance("Set Number of Readout blocks Y direction");

	cmdName10 = GetDirectoryName() + "setNumberReadOutBlocksZ";
	pNumberReadOutBlocksZ =  new G4UIcmdWithAnInteger(cmdName10,this);
	pNumberReadOutBlocksZ->SetGuidance("Set Number of Readout blocks Z direction");


}


GateDiscretizationMessenger::~GateDiscretizationMessenger()
{
	delete  DiscCmd;
	delete pStripOffsetX;
	delete pStripOffsetY;
	delete pStripOffsetZ;
	delete pStripWidthX;
	delete pStripWidthY;
	delete pStripWidthZ;
	delete pNumberStripsX;
	delete pNumberStripsY;
	delete pNumberStripsZ;
	delete pNumberReadOutBlocksY;
	delete pNumberReadOutBlocksX;
	delete pNumberReadOutBlocksZ;
}

void GateDiscretizationMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{


	    if ( command== pStripOffsetX ) {
			m_GateDiscretization->SetStripOffsetX(m_name, pStripOffsetX->GetNewDoubleValue(newValue));

	    }

		if ( command==pStripOffsetY ) {
			m_GateDiscretization->SetStripOffsetY(m_name, pStripOffsetY->GetNewDoubleValue(newValue));


		}

		if ( command==pStripOffsetZ ) {
			m_GateDiscretization->SetStripOffsetZ(m_name, pStripOffsetZ->GetNewDoubleValue(newValue));


		}

		if ( command==pStripWidthX ) {
			m_GateDiscretization->SetStripWidthX(m_name, pStripWidthX->GetNewDoubleValue(newValue));


		}



		if ( command==pStripWidthY ) {
			m_GateDiscretization->SetStripWidthY(m_name, pStripWidthY->GetNewDoubleValue(newValue));


		}

		if ( command==pStripWidthZ ) {
			m_GateDiscretization->SetStripWidthZ(m_name, pStripWidthZ->GetNewDoubleValue(newValue));


		}

		if ( command== pNumberStripsX ) {
			m_GateDiscretization->SetNumberStripsX(m_name, pNumberStripsX->GetNewIntValue(newValue));


		}

		if ( command== pNumberStripsY ) {
			m_GateDiscretization->SetNumberStripsY(m_name, pNumberStripsY->GetNewIntValue(newValue));

		}


		if ( command== pNumberStripsZ ) {
			m_GateDiscretization->SetNumberStripsZ(m_name, pNumberStripsZ->GetNewIntValue(newValue));

		}


		if ( command== pNumberReadOutBlocksX ) {
			m_GateDiscretization->SetNumberReadOutBlocksX(m_name, pNumberReadOutBlocksX->GetNewIntValue(newValue));

		}


		if ( command== pNumberReadOutBlocksY ) {
			m_GateDiscretization->SetNumberReadOutBlocksY(m_name, pNumberReadOutBlocksY->GetNewIntValue(newValue));

		}


		if ( command== pNumberReadOutBlocksZ ) {
			m_GateDiscretization->SetNumberReadOutBlocksZ(m_name, pNumberReadOutBlocksZ->GetNewIntValue(newValue));

		}

		else
		{
			GateClockDependentMessenger::SetNewValue(command,newValue);
		}



}


