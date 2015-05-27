/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateTotalDiscreteProcess.hh"
#include "G4SystemOfUnits.hh"
#include "G4ExceptionHandler.hh"
#include <cassert>
#include "G4Material.hh"
#include "G4VEmProcess.hh"
#include "GateCrossSectionsTable.hh"
#include <fstream>
#include "GatePETVRTManager.hh"
#include "GatePETVRTSettings.hh"
#include "GateMessageManager.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

using namespace std;

GateTotalDiscreteProcess::GateTotalDiscreteProcess ( const G4String& name, G4ProcessType type, G4int num, const G4ParticleDefinition* p, G4double minn, G4double maxx, G4int binn )
		:G4VDiscreteProcess ( name, type ),pParticleType ( p ),m_oProcessVec ( num )
{
	assert ( *p==*G4Gamma::GammaDefinition() );
	m_nNumProcesses=0;
	m_nMaxNumProcesses=num;
	m_nTotalMinEnergy =minn;
	m_nTotalMaxEnergy =maxx;
	m_pTotalCrossSectionsTable=NULL;
	m_nTotalBinNumber=binn;
}


GateTotalDiscreteProcess::~GateTotalDiscreteProcess()
{
	for (Gate_ProcessVec::iterator it=m_oProcessVec.begin();it!=m_oProcessVec.end(); )
	{
	  it=m_oProcessVec.erase(it);
	}
	if ( m_pTotalCrossSectionsTable!=NULL ) delete m_pTotalCrossSectionsTable;
}


bool GateTotalDiscreteProcess::AddDiscreteProcess ( G4VDiscreteProcess* p )
{
	if ( p==NULL )
	{
		G4Exception ( "GateTotalDiscreteProcess::AddDiscreteProcess (const G4VDiscreteProcess*)", "InvalidSetup", FatalException,"NULL pointer as parameter!" );
		return false;
	}

	if ( !p->IsApplicable ( *pParticleType ) )
	{
		G4Exception ( "GateTotalDiscreteProcess::AddDiscreteProcess (const G4VDiscreteProcess*)", "InvalidSetup", FatalException, G4String ( "Added DiscreteProcess that is not compatible to " + pParticleType->GetParticleName() +"!" ).c_str() );
		return false;
	}
	

	if ( static_cast<G4int> ( m_oProcessVec.size() ) <=m_nMaxNumProcesses )
	{
		m_oProcessVec[m_nNumProcesses].Process = p;
		m_oProcessVec[m_nNumProcesses].Name = new G4String ( p->GetProcessName() );
		m_nNumProcesses++;
		return true;
	}

	G4Exception ( "GateTotalDiscreteProcess::AddDiscreteProcess (const G4VDiscreteProcess*)", "InvalidSetup", FatalException,"More processes added than specified in constructor! This is most likely a bug." );
	return false;
}

void 	GateTotalDiscreteProcess::BuildPhysicsTable ( const G4ParticleDefinition & part )
{
#ifdef G4VERBOSE
	G4cout << "Build physics table for GateTotalDiscreteProcess.\n";
#endif

	if ( part!=*pParticleType )
	{
		G4Exception ( "GateTotalDiscreteProcess::BuildPhysicsTable(const G4ParticleDefinition&)", "InvalidSetup", FatalException,G4String ( "Particle not valid for process '"+GetProcessName() +"'!" ).c_str() );
	}
	assert ( m_nNumProcesses==static_cast<G4int> ( m_oProcessVec.size() ) );
	if ( m_nMaxNumProcesses!=m_nNumProcesses )
	{
		G4Exception ( "GateTotalDiscreteProcess::BuildPhysicsTable(const G4ParticleDefinition&)", "InvalidSetup", FatalException,"Not enough processes added! This is most likely a bug." );
	}

	BuildCrossSectionsTables();
	vector<G4Material*> vec;
	if ( GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetFictitiousMap() !=NULL )
	{
		GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetFictitiousMap()->GetMaterials ( vec );
		CreateTotalMaxCrossSectionTable ( vec );
	}
	else
	{
//		G4Exception ("Error! 'Fictitious' without 'fictitiousVoxelMap' is not valid!", "InvalidSetup", FatalException,"Remove selectFictitious or add fictitiousVoxelMap" );
		GateWarning("Warning: The 'Fictitious' process is used without using a 'fictitiousVoxelMap' as geometry !\nAll gamma processes are forced.");
	}

}

void GateTotalDiscreteProcess::BuildCrossSectionsTables()
{
	// build tables for single processes
        std::vector<G4VDiscreteProcess*> vec;
	for (Gate_ProcessVec::iterator it=m_oProcessVec.begin();it!=m_oProcessVec.end(); ++it)
	{
		it->Process->PreparePhysicsTable ( *pParticleType );
		it->Process->BuildPhysicsTable ( *pParticleType );
		it->CrossSectionsTable=new GateCrossSectionsTable ( m_nTotalMinEnergy,m_nTotalMaxEnergy,m_nTotalBinNumber,pParticleType,*(it->Process) );

#ifdef G4VERBOSE
		G4cout << "***************\n";
		G4cout << "GATE SUBPROCESS " << *(it->Name) <<" : Building fast linear tables for "<< pParticleType->GetParticleName() << " in the energy range [" << m_nTotalMinEnergy/keV << "," << m_nTotalMaxEnergy/keV << "] keV in " << m_nTotalBinNumber << " " << ( m_nTotalMaxEnergy-m_nTotalMinEnergy ) /m_nTotalBinNumber/keV << " keV bins\n";
		G4cout << "***************\n";
#endif
		it->CrossSectionsTable->SetAndBuildProductionMaterialTable();
		vec.push_back ( it->Process );
	}

	// build tables for total cross section
	m_pTotalCrossSectionsTable=new GateCrossSectionsTable ( m_nTotalMinEnergy,m_nTotalMaxEnergy,m_nTotalBinNumber,pParticleType,vec);
#ifdef G4VERBOSE
	G4cout << "*****************\n";
	G4cout << "GATE TOTALPROCESS " << GetProcessName() <<" : Building fast linear tables for "<< pParticleType->GetParticleName() << " in the energy range [" << m_nTotalMinEnergy/keV << "," << m_nTotalMaxEnergy/keV << "] keV in " << m_nTotalBinNumber << " " << ( m_nTotalMaxEnergy-m_nTotalMinEnergy ) /m_nTotalBinNumber/keV << " keV bins\n";
	G4cout << "*****************\n";
#endif
	m_pTotalCrossSectionsTable->SetAndBuildProductionMaterialTable();
}


G4VParticleChange * 	GateTotalDiscreteProcess::PostStepDoIt ( const G4Track &track, const G4Step &stepData )
{
	G4VParticleChange * returnvalue=m_oProcessVec[m_nProcessWithSmallestPIL].Process-> PostStepDoIt ( track,stepData );
	track.GetStep()->GetPostStepPoint()->SetProcessDefinedStep ( m_oProcessVec[m_nProcessWithSmallestPIL].Process );
	return returnvalue;
}

const GateCrossSectionsTable* GateTotalDiscreteProcess::GetTotalCrossSectionsTable() const
{
	return m_pTotalCrossSectionsTable;
}


G4double GateTotalDiscreteProcess::GetMeanFreePath ( const G4Track &aTrack, G4double , G4ForceCondition *)
{
	return ( 1./m_pTotalCrossSectionsTable->GetCrossSection ( aTrack.GetMaterial(), aTrack.GetKineticEnergy() ) );
}

void GateTotalDiscreteProcess::CreateTotalMaxCrossSectionTable ( const std::vector<G4Material*>& vec )
{
#ifdef G4VERBOSE
	G4cout << "******************************\n";
	G4cout << "GATE TOTALMAXFICTITIOUSPROCESS "<< GetProcessName() << " : Building fast linear tables for "<< pParticleType->GetParticleName() << " in the energy range [" << m_nTotalMinEnergy/keV << "," << m_nTotalMaxEnergy/keV << "] keV in " << m_nTotalBinNumber << " " << ( m_nTotalMaxEnergy-m_nTotalMinEnergy ) /m_nTotalBinNumber/keV << " keV bins\n";
	G4cout << "******************************\n";
#endif
	m_pTotalCrossSectionsTable->BuildMaxCrossSection ( vec );
}




