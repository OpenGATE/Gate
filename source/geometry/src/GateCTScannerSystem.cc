/*----------------------
     
   Didier Benoit <benoit@cppm.in2p3.fr>
   Franca Cassol Brunner <cassol@cppm.in2p3.fr>
     
   Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

   This software is distributed under the terms 
   of the GNU Lesser General  Public Licence (LGPL) 
   See GATE/LICENSE.txt for further details 
----------------------*/

#include "GateCTScannerSystem.hh"
#include "GateToImageCT.hh"
#include "GateClockDependentMessenger.hh"
#include "GateArrayComponent.hh"
#include "GateOutputMgr.hh"

GateCTScannerSystem::GateCTScannerSystem( const G4String& itsName )
	: GateVSystem( itsName, true )
{
	//Set up a messenger
	m_messenger = new GateClockDependentMessenger( this );
	m_messenger->SetDirectoryGuidance( G4String( "Controls the system '" ) +
		GetObjectName() + "'" );
	
	//Define the CT scanner components
	//CTScanner
	GateArrayComponent* moduleComponent =
		new GateArrayComponent( "module", GetBaseComponent(), this );
	GateArrayComponent* clusterComponent_0 =
		new GateArrayComponent( "cluster_0", moduleComponent, this );
		new GateArrayComponent( "pixel_0", clusterComponent_0, this );
	GateArrayComponent* clusterComponent_1 =
		new GateArrayComponent( "cluster_1", moduleComponent, this );
		new GateArrayComponent( "pixel_1", clusterComponent_1, this );
	GateArrayComponent* clusterComponent_2 =
		new GateArrayComponent( "cluster_2", moduleComponent, this );
		new GateArrayComponent( "pixel_2", clusterComponent_2, this );
	
	SetOutputIDName((char *)"gantryID", 0);
	SetOutputIDName((char *)"moduleID", 1);
	SetOutputIDName((char *)"clusterID", 2);
	SetOutputIDName((char *)"pixelID", 3);
	
	GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
	m_gateToImageCT = new GateToImageCT("imageCT", outputMgr,
		this, GateOutputMgr::GetDigiMode() );
	outputMgr->AddOutputModule((GateVOutputModule*)m_gateToImageCT);
}

GateCTScannerSystem::~GateCTScannerSystem()
{
	delete m_messenger;
}
