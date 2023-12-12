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

#include "GateGridDiscretizatorMessenger.hh"
#include "GateGridDiscretizator.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"



GateGridDiscretizatorMessenger::GateGridDiscretizatorMessenger (GateGridDiscretizator* GateGridDiscretizator)
:GateClockDependentMessenger(GateGridDiscretizator),
 	 m_GateGridDiscretizator(GateGridDiscretizator)
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


GateGridDiscretizatorMessenger::~GateGridDiscretizatorMessenger()
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

void GateGridDiscretizatorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{


	    if ( command== pStripOffsetX ) {
			m_GateGridDiscretizator->SetStripOffsetX( pStripOffsetX->GetNewDoubleValue(newValue));

	    }

	    else if ( command==pStripOffsetY ) {
			m_GateGridDiscretizator->SetStripOffsetY( pStripOffsetY->GetNewDoubleValue(newValue));


		}

	    else if ( command==pStripOffsetZ ) {
			m_GateGridDiscretizator->SetStripOffsetZ( pStripOffsetZ->GetNewDoubleValue(newValue));


		}

	    else if ( command==pStripWidthX ) {
			m_GateGridDiscretizator->SetStripWidthX( pStripWidthX->GetNewDoubleValue(newValue));


		}



	    else if ( command==pStripWidthY ) {
			m_GateGridDiscretizator->SetStripWidthY( pStripWidthY->GetNewDoubleValue(newValue));


		}

	    else if ( command==pStripWidthZ ) {
			m_GateGridDiscretizator->SetStripWidthZ( pStripWidthZ->GetNewDoubleValue(newValue));


		}

	    else if ( command== pNumberStripsX ) {
			m_GateGridDiscretizator->SetNumberStripsX( pNumberStripsX->GetNewIntValue(newValue));


		}

	    else if ( command== pNumberStripsY ) {
			m_GateGridDiscretizator->SetNumberStripsY( pNumberStripsY->GetNewIntValue(newValue));

		}


	    else if ( command== pNumberStripsZ ) {
			m_GateGridDiscretizator->SetNumberStripsZ( pNumberStripsZ->GetNewIntValue(newValue));

		}


	    else if ( command== pNumberReadOutBlocksX ) {
			m_GateGridDiscretizator->SetNumberReadOutBlocksX( pNumberReadOutBlocksX->GetNewIntValue(newValue));

		}


	    else if ( command== pNumberReadOutBlocksY ) {
			m_GateGridDiscretizator->SetNumberReadOutBlocksY( pNumberReadOutBlocksY->GetNewIntValue(newValue));

		}


	    else if ( command== pNumberReadOutBlocksZ ) {
			m_GateGridDiscretizator->SetNumberReadOutBlocksZ( pNumberReadOutBlocksZ->GetNewIntValue(newValue));

		}

		else
		{
			GateClockDependentMessenger::SetNewValue(command,newValue);
		}



}


