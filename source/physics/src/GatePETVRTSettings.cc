/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GatePETVRTSettings.hh"
#include "GateFictitiousFastSimulationModel.hh"
#include "GateTotalDiscreteProcess.hh"
#include "G4ios.hh"
#include "GatePhantomSD.hh"

GatePETVRTSettings::GatePETVRTSettings()
{
	m_nDeleteFictitiousMap=false;
	m_nDeleteTotalDiscreteProcess=false;
	m_nDeleteFictitiousFastSimulationModel=false;
	m_nDeletePhantomSD=false;
	pFictitiousMap=NULL;
	pEnvelope=NULL;
	m_nApproximations=GatePETVRT::kNotSet;
	pFictitiousFastSimulationModel=NULL;
	pTotalDiscreteProcess=NULL;
	pPhantomSD=NULL;
	m_nFictitiousEnergy=-1;
	m_nDiscardEnergy=-1;
	m_nVerbosityLevel=Verbose;
}


GatePETVRTSettings::~GatePETVRTSettings()
{
	if ( m_nDeleteFictitiousMap ) delete pFictitiousMap;
	if ( m_nDeleteTotalDiscreteProcess ) delete pTotalDiscreteProcess;
	if ( m_nDeleteFictitiousFastSimulationModel ) delete pFictitiousFastSimulationModel;
	if ( m_nDeletePhantomSD) delete pPhantomSD;
}


void GatePETVRTSettings::RegisterEnvelope ( G4Envelope* p )
{
	pEnvelope=p;
}

void GatePETVRTSettings::SetFictitiousEnergy ( G4double en )
{
	m_nFictitiousEnergy=en;
	if (m_nDiscardEnergy>=0)
	{

		if (m_nVerbosityLevel>=Verbose)
		{
			G4cout << "GatePETVRTSettings::SetFictitiousEnergy: Set to "<< m_nFictitiousEnergy << G4endl;
		}
	}
}

void GatePETVRTSettings::SetDiscardEnergy ( G4double en )
{
	m_nDiscardEnergy=en;
	if (m_nFictitiousEnergy>=0)
	{
	
		if (m_nVerbosityLevel>=Verbose)
		{
			G4cout << "GatePETVRTSettings::SetDiscardEnergy: Set to "<< m_nDiscardEnergy << G4endl;
		}
	}
}
void GatePETVRTSettings::RegisterFictitiousMap ( GateVFictitiousMap* map, bool deleteWithThis )
{
	if ( ( pFictitiousMap !=NULL ) && ( m_nDeleteFictitiousMap ) )
	{
		G4Exception ( "GatePETVRTSettings::RegisterFictitiousMap(GateVFictitiousMap*, bool)", "FictitiousMap already defined", FatalException,
		              "pFictitiousMap!=NULL!" );
	}
	m_nDeleteFictitiousMap=deleteWithThis;
	pFictitiousMap=map;
}

void GatePETVRTSettings::RegisterTotalDiscreteProcess ( GateTotalDiscreteProcess* proc, bool deleteWithThis )
{
	if ( ( pTotalDiscreteProcess!=NULL ) && ( m_nDeleteTotalDiscreteProcess ) )
	{
		G4Exception ( "GatePETVRTSettings::RegisterDiscreteProcess(GateTotalDiscreteProcess* proc, bool)", "TotalDiscreteProcess already defined", FatalException,
		              "pTotalDiscreteProcess!=NULL!" );
	}
	m_nDeleteTotalDiscreteProcess=deleteWithThis;
	pTotalDiscreteProcess=proc;
}

void GatePETVRTSettings::RegisterPhantomSD(GatePhantomSD* p, bool deleteWithThis)
{
		if ( ( pPhantomSD!=NULL ) && ( m_nDeletePhantomSD ) )
{
		G4Exception ( "GatePETVRTSettings::RegisterPhantomSD(GatePhantomSD*, bool)", "PhantomSD already registered", FatalException,
		              "pPhantomSD!=NULL!" );
	}
	m_nDeletePhantomSD=deleteWithThis;
	pPhantomSD=p;
}



void GatePETVRTSettings::SetApproximations ( GatePETVRT::Approx a )
{
	m_nApproximations= a;
}

void GatePETVRTSettings::RegisterFictitiousFastSimulationModel ( GateFictitiousFastSimulationModel* model, bool deleteWithThis )
{
	if ( ( pFictitiousFastSimulationModel!=NULL ) && ( m_nDeleteFictitiousFastSimulationModel ) )
	{
		G4Exception ( "GatePETVRTSettings::RegisterFictitiousFastSimulationModel(GateFictitiousFastSimulationModel* model, bool deleteWithThis)", "FictitiousFastSimulationModel already defined", FatalException,
		              "pFictitiousFastSimulationModel!=NULL!" );
	}
	m_nDeleteFictitiousFastSimulationModel=deleteWithThis;
	pFictitiousFastSimulationModel=model;

}
GateFictitiousFastSimulationModel* GatePETVRTSettings::GetFictitiousFastSimulationModel()
{
	return pFictitiousFastSimulationModel;
}

GateTotalDiscreteProcess* GatePETVRTSettings::GetTotalDiscreteProcess() const
{
	return pTotalDiscreteProcess;
}
