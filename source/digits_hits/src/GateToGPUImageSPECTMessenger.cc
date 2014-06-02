/*----------------------
   OpenGATE Collaboration

   Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

#include "GateToGPUImageSPECTMessenger.hh"
#include "GateToGPUImageSPECT.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

GateToGPUImageSPECTMessenger::GateToGPUImageSPECTMessenger( GateToGPUImageSPECT* GateToGPUImageSPECT )
	: GateOutputModuleMessenger( GateToGPUImageSPECT ),
	m_gateToGPUImageSPECT( GateToGPUImageSPECT )
{
	G4String cmdName;

	cmdName = GetDirectoryName() + "setFileName";
	setFileNameCmd = new G4UIcmdWithAString( cmdName, this );
	setFileNameCmd->SetGuidance( "Set the name of the output image files" );
	setFileNameCmd->SetParameterName( "Name", false );

    cmdName = GetDirectoryName() + "attachTo";
	attachToCmd = new G4UIcmdWithAString( cmdName, this );
	attachToCmd->SetGuidance( "Set the name of volume to attach" );
	attachToCmd->SetParameterName( "Name", false );

    cmdName = GetDirectoryName() + "bufferParticleEntry";
    bufferParticleEntryCmd = new G4UIcmdWithAnInteger( cmdName, this );
    bufferParticleEntryCmd->SetGuidance( "Set the buffer particle size" );
	bufferParticleEntryCmd->SetParameterName( "Buffer size", false );

    cmdName = GetDirectoryName() + "cudaDevice";
    cudaDeviceCmd = new G4UIcmdWithAnInteger( cmdName, this );
    cudaDeviceCmd->SetGuidance( "Set the cuda device" );
	cudaDeviceCmd->SetParameterName( "CUDA device", false );

		cmdName = GetDirectoryName() + "cpuNumber";
    cpuNumberCmd = new G4UIcmdWithAnInteger( cmdName, this );
    cpuNumberCmd->SetGuidance( "Set the number of thread cpus" );
    cpuNumberCmd->SetParameterName( "CPU device", false );

    cmdName = GetDirectoryName() + "setCPUflag";
    cpuFlagCmd = new G4UIcmdWithABool( cmdName, this );
    cpuFlagCmd->SetGuidance( "Set the cpu flag" );
    cpuFlagCmd->SetParameterName( "cpu flag", false );

    cmdName = GetDirectoryName() + "setRootHitFlag";
    rootHitCmd = new G4UIcmdWithABool( cmdName, this );
    rootHitCmd->SetGuidance( "Set the root hit flag" );
	rootHitCmd->SetParameterName( "root hit flag", false );

    cmdName = GetDirectoryName() + "setRootSingleFlag";
    rootSingleCmd = new G4UIcmdWithABool( cmdName, this );
    rootSingleCmd->SetGuidance( "Set the root single flag" );
	rootSingleCmd->SetParameterName( "root single flag", false );

	cmdName = GetDirectoryName() + "setRootSourceFlag";
	rootSourceCmd = new G4UIcmdWithABool( cmdName, this );
	rootSourceCmd->SetGuidance( "Set the root source flag" );
	rootSourceCmd->SetParameterName( "root source flag", false );

	cmdName = GetDirectoryName() + "setRootExitCollimatorSourceFlag";
	rootExitCollimatorSourceCmd = new G4UIcmdWithABool( cmdName, this );
	rootExitCollimatorSourceCmd->SetGuidance(
		"Set the root exit collimator source flag" );
	rootExitCollimatorSourceCmd->SetParameterName(
		"root exit collimator source flag", false );

	cmdName = GetDirectoryName() + "timeFlag";
	timeCmd = new G4UIcmdWithABool( cmdName, this );
	timeCmd->SetGuidance( "Set the time flag" );
	timeCmd->SetParameterName( "time flag", false );

    cmdName = GetDirectoryName() + "setNZpixel";
    nzPixelCmd = new G4UIcmdWithAnInteger( cmdName, this );
    nzPixelCmd->SetGuidance( "Set the number of pixel in Z" );
	nzPixelCmd->SetParameterName( "number of pixel in Z", false );

    cmdName = GetDirectoryName() + "setNYpixel";
    nyPixelCmd = new G4UIcmdWithAnInteger( cmdName, this );
    nyPixelCmd->SetGuidance( "Set the number of pixel in Y" );
	nyPixelCmd->SetParameterName( "number of pixel in Y", false );

		cmdName = GetDirectoryName() + "setZpixelSize";
    zPixelSizeCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    zPixelSizeCmd->SetGuidance( "Set the size of pixel in Z" );
    zPixelSizeCmd->SetUnitCategory("Length");

		cmdName = GetDirectoryName() + "setYpixelSize";
    yPixelSizeCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    yPixelSizeCmd->SetGuidance( "Set the size of pixel in Y" );
    yPixelSizeCmd->SetUnitCategory("Length");


    cmdName = GetDirectoryName() + "setSepta";
    septaCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    septaCmd->SetGuidance( "Set the size of septa" );
    septaCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName() + "setCollimatorHeight";
    collimatorHeightCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    collimatorHeightCmd->SetGuidance( "Set the size of collimateur height" );
    collimatorHeightCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName() + "setFy";
    fyCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    fyCmd->SetGuidance( "Set the focal in Y" );
    fyCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName() + "setFz";
    fzCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    fzCmd->SetGuidance( "Set the focal in Z" );
    fzCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName() + "setSpaceBetweenCollimatorDetector";
    spaceBetweenCollimatorDetectorCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    spaceBetweenCollimatorDetectorCmd->SetGuidance( "Set the size of septa" );
    spaceBetweenCollimatorDetectorCmd->SetUnitCategory("Length");

		cmdName = GetDirectoryName() + "setRor";
    rorCmd = new G4UIcmdWithADoubleAndUnit( cmdName, this );
    rorCmd->SetGuidance( "Set the radius of rotation (ROR)" );
    rorCmd->SetUnitCategory("Length");
}

GateToGPUImageSPECTMessenger::~GateToGPUImageSPECTMessenger()
{
	delete setFileNameCmd;
	delete attachToCmd;
	delete bufferParticleEntryCmd;
	delete cudaDeviceCmd;
	delete rootHitCmd;
	delete rootSingleCmd;
	delete rootSourceCmd;
	delete rootExitCollimatorSourceCmd;
	delete timeCmd;
	delete nzPixelCmd;
	delete nyPixelCmd;
	delete septaCmd;
	delete fyCmd;
	delete fzCmd;
	delete zPixelSizeCmd;
	delete yPixelSizeCmd;
	delete collimatorHeightCmd;
	delete spaceBetweenCollimatorDetectorCmd;
	delete rorCmd;
	delete cpuNumberCmd;
	delete cpuFlagCmd;
}

void GateToGPUImageSPECTMessenger::SetNewValue( G4UIcommand* command,
	G4String newValue )
{
	if( command == setFileNameCmd )
		m_gateToGPUImageSPECT->SetFileName( newValue );
    else if( command == attachToCmd )
        m_gateToGPUImageSPECT->SetVolumeToAttach( newValue );
    else if( command == bufferParticleEntryCmd )
        m_gateToGPUImageSPECT->SetBufferParticleEntry(
            bufferParticleEntryCmd->GetNewIntValue( newValue ) );
    else if( command == cudaDeviceCmd )
        m_gateToGPUImageSPECT->SetCudaDevice(
            cudaDeviceCmd->GetNewIntValue( newValue ) );
		else if( command == cpuNumberCmd )
			m_gateToGPUImageSPECT->SetCpuNumber( cpuNumberCmd->GetNewIntValue( newValue ) );
		else if( command == cpuFlagCmd )
			m_gateToGPUImageSPECT->SetCpuFlag( cpuFlagCmd->GetNewBoolValue( newValue ) );
    else if( command == rootHitCmd )
        m_gateToGPUImageSPECT->SetRootHitFlag(
            rootHitCmd->GetNewBoolValue( newValue ) );
    else if( command == rootSingleCmd )
        m_gateToGPUImageSPECT->SetRootSingleFlag(
            rootSingleCmd->GetNewBoolValue( newValue ) );
    else if( command == rootSourceCmd )
        m_gateToGPUImageSPECT->SetRootSourceFlag(
            rootSourceCmd->GetNewBoolValue( newValue ) );
		else if( command == rootExitCollimatorSourceCmd )
        m_gateToGPUImageSPECT->SetRootExitCollimatorSourceFlag(
            rootExitCollimatorSourceCmd->GetNewBoolValue( newValue ) );
		else if( command == timeCmd )
        m_gateToGPUImageSPECT->SetTimeFlag(
            timeCmd->GetNewBoolValue( newValue ) );
    else if( command == nzPixelCmd )
        m_gateToGPUImageSPECT->SetNZpixel(
            nzPixelCmd->GetNewIntValue( newValue ) );
    else if( command == nyPixelCmd )
        m_gateToGPUImageSPECT->SetNYpixel(
            nyPixelCmd->GetNewIntValue( newValue ) );
		else if( command == zPixelSizeCmd )
			m_gateToGPUImageSPECT->SetZPixelSize(
				zPixelSizeCmd->GetNewDoubleValue( newValue ) );
		else if( command == yPixelSizeCmd )
			m_gateToGPUImageSPECT->SetYPixelSize(
				yPixelSizeCmd->GetNewDoubleValue( newValue ) );
    else if( command == septaCmd )
        m_gateToGPUImageSPECT->SetSepta(
            septaCmd->GetNewDoubleValue( newValue ) );
    else if( command == fyCmd )
        m_gateToGPUImageSPECT->SetFy(
            fyCmd->GetNewDoubleValue( newValue ) );
    else if( command == fzCmd )
        m_gateToGPUImageSPECT->SetFz(
            fzCmd->GetNewDoubleValue( newValue ) );
    else if( command == collimatorHeightCmd )
        m_gateToGPUImageSPECT->SetCollimatorHeight(
            collimatorHeightCmd->GetNewDoubleValue( newValue ) );
    else if( command == spaceBetweenCollimatorDetectorCmd )
        m_gateToGPUImageSPECT->SetSpaceBetweenCollimatorDetector(
            spaceBetweenCollimatorDetectorCmd->GetNewDoubleValue( newValue ) );
		else if( command == rorCmd )
        m_gateToGPUImageSPECT->SetRor( rorCmd->GetNewDoubleValue( newValue ) );
	else
		GateOutputModuleMessenger::SetNewValue( command, newValue );
}
