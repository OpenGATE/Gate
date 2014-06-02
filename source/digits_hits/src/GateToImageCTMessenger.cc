/*----------------------
   OpenGATE Collaboration

   Didier Benoit <benoit@cppm.in2p3.fr>
   Franca Cassol Brunner <cassol@cppm.in2p3.fr>

   Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

#include "GateToImageCTMessenger.hh"
#include "GateToImageCT.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

GateToImageCTMessenger::GateToImageCTMessenger( GateToImageCT* GateToImageCT )
	: GateOutputModuleMessenger( GateToImageCT ),
	m_gateToImageCT( GateToImageCT )
{
	G4String cmdName;

	cmdName = GetDirectoryName() + "setFileName";
	setFileNameCmd = new G4UIcmdWithAString( cmdName, this );
	setFileNameCmd->SetGuidance( "Set the name of the output image files" );
	setFileNameCmd->SetParameterName( "Name", false );

/* Forbidden now !! Only the mother inherited enable and disable commands are allowed
	cmdName = GetDirectoryName() + "rawOutputEnable";
	rawOutputCmd = new G4UIcmdWithABool( cmdName, this );
	rawOutputCmd->SetGuidance( "Enables CT-Images output in a raw file" );
*/
	cmdName = GetDirectoryName() + "numFastPixelX";
	numFastPixelXCmd = new G4UIcmdWithAnInteger( cmdName, this );
	numFastPixelXCmd->SetGuidance( "Number of pixels in X" );

	cmdName = GetDirectoryName() + "numFastPixelY";
	numFastPixelYCmd = new G4UIcmdWithAnInteger( cmdName, this );
	numFastPixelYCmd->SetGuidance( "Number of pixels in Y" );

	cmdName = GetDirectoryName() + "numFastPixelZ";
	numFastPixelZCmd = new G4UIcmdWithAnInteger( cmdName, this );
	numFastPixelZCmd->SetGuidance( "Number of pixels in Z" );

	cmdName = GetDirectoryName() + "setStartSeed";
	setStartSeedCmd = new G4UIcmdWithAnInteger( cmdName, this );
	setStartSeedCmd->SetGuidance( "Set the starting random Seed" );
	setStartSeedCmd->SetParameterName( "Seed", false );

	cmdName = GetDirectoryName() + "vrtFactor";
	vrtFactorCmd = new G4UIcmdWithAnInteger( cmdName, this );
  	vrtFactorCmd->SetGuidance( "Number of self created photons" );

	cmdName = GetDirectoryName() + "detX";
	detectorInXCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
  	detectorInXCmd->SetGuidance("Det X");
	detectorInXCmd->SetDefaultUnit( "mm" );

	cmdName = GetDirectoryName() + "detY";
	detectorInYCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
  	detectorInYCmd->SetGuidance("Det Y");
	detectorInYCmd->SetDefaultUnit( "mm" );

	cmdName = GetDirectoryName() + "sourceDetector";
	sourceDetectorCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
  	sourceDetectorCmd->SetGuidance("distance Source-Detector");
	sourceDetectorCmd->SetDefaultUnit( "mm" );
}

GateToImageCTMessenger::~GateToImageCTMessenger()
{
	delete setFileNameCmd;
//	delete rawOutputCmd;
	delete setStartSeedCmd;
	delete vrtFactorCmd;
	delete numFastPixelXCmd;
	delete numFastPixelYCmd;
	delete numFastPixelZCmd;
	delete detectorInXCmd;
	delete detectorInYCmd;
	delete sourceDetectorCmd;
}

void GateToImageCTMessenger::SetNewValue( G4UIcommand* command,
	G4String newValue )
{
	if( command == setFileNameCmd )
		m_gateToImageCT->SetFileName( newValue );
/*	else if( command == rawOutputCmd )
		m_gateToImageCT->Enable( rawOutputCmd->GetNewBoolValue( newValue ) );*/
	else if( command == setStartSeedCmd )
		m_gateToImageCT->SetStartSeed(
			setStartSeedCmd->GetNewIntValue( newValue ) );
	else if( command == vrtFactorCmd )
		m_gateToImageCT->SetVRTFactor(
			vrtFactorCmd->GetNewIntValue( newValue ) );
	else if( command == numFastPixelXCmd )
		m_gateToImageCT->SetFastPixelXNb(
			numFastPixelXCmd->GetNewIntValue( newValue ) );
	else if( command == numFastPixelYCmd )
		m_gateToImageCT->SetFastPixelYNb(
			numFastPixelYCmd->GetNewIntValue( newValue ) );
	else if( command == numFastPixelZCmd )
		m_gateToImageCT->SetFastPixelZNb(
			numFastPixelZCmd->GetNewIntValue( newValue ) );
	 else if ( command == detectorInXCmd )
		m_gateToImageCT->SetDetectorX( detectorInXCmd->GetNewDoubleValue(
			newValue ) );
	else if ( command == detectorInYCmd )
    		m_gateToImageCT->SetDetectorY( detectorInYCmd->GetNewDoubleValue(
			newValue ) );
	else if( command == sourceDetectorCmd )
		m_gateToImageCT->SetSourceDetector( sourceDetectorCmd
			->GetNewDoubleValue( newValue ) );
	else
		GateOutputModuleMessenger::SetNewValue( command, newValue );
}
