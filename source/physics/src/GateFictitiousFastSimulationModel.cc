/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateFictitiousFastSimulationModel.hh"

#include "GateFictitiousVoxelMap.hh"
#include "GateCrossSectionsTable.hh"
#include <cassert>
#include "G4FastTrack.hh"
#include "Randomize.hh"
#include "G4AffineTransform.hh"
#include "GateTotalDiscreteProcess.hh"
#include "GatePETVRTManager.hh"
#include "G4FastSimulationManager.hh"
#include "G4TouchableHistory.hh"
#include "GatePhantomSD.hh"
#include "G4FastStep.hh"
#include "globals.hh"

using namespace std;


GateFictitiousFastSimulationModel::GateFictitiousFastSimulationModel ( G4double minEnergy, G4double maxEnergy )
		: G4VFastSimulationModel ( "Fictitious interaction model" ),pTotalCrossSectionsTable ( NULL ),pFictitiousMap ( NULL ),pTotalDiscreteProcess ( NULL ),  m_nApproximations ( GatePETVRT::kVolumeTrace ), pCurrentFastTrack ( NULL ),pCurrentFastStep ( NULL ),pPhantomSD ( NULL ) //, pMaxMaterial(NULL)
{
	m_nAbsMinEnergy=minEnergy;
	m_nAbsMaxEnergy=maxEnergy;
	m_nMinEnergy=minEnergy;
	m_nMaxEnergy=maxEnergy;
	m_nInitialized=false;
	m_nCurrentInvFictCrossSection=-1; // not initialized
	m_pTrackFastVector=new G4TrackFastVector();
	m_nNumSecondaries=0;
	m_nSurfaceTolerance=G4GeometryTolerance::GetInstance()->GetSurfaceTolerance() *3.;
}


GateFictitiousFastSimulationModel::~GateFictitiousFastSimulationModel()
{
	delete m_pTrackFastVector;
}


G4bool GateFictitiousFastSimulationModel::IsApplicable ( const G4ParticleDefinition& def )
{
	if ( G4Gamma::GammaDefinition() ==&def ) return true;
	return false;
}

G4bool GateFictitiousFastSimulationModel::ModelTrigger ( const G4FastTrack & ft )
{
	G4double en=ft.GetPrimaryTrack()->GetKineticEnergy();
	return ( ( en>m_nMinEnergy ) && ( en<m_nMaxEnergy ) );
}


void GateFictitiousFastSimulationModel::DoIt ( const G4FastTrack& ft, G4FastStep& fs )
{
	if ( !m_nInitialized )
	{
		if ( !SetParameters ( GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings() ) )
		{
			G4Exception ( "GateFictitiousFastSimulationModel::DoIt(const G4FastTrack&, G4FastStep&)", "Not all necessary things initialized", FatalException,
			              "FictitiousMap or TotalDiscreteProcess or TotalCrossSection not set " );
		}
		pFictitiousMap->RegisterCrossSectionsTable ( GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetTotalDiscreteProcess()->GetTotalCrossSectionsTable(),false );
		m_nInitialized=true;
		pTotalCrossSectionsTable=GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetTotalDiscreteProcess()->GetTotalCrossSectionsTable();

		m_nDiscardEnergy=GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetDiscardEnergy();

		if ( m_nDiscardEnergy<=0 )
		{
			if ( GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetFictitiousEnergy() <=0 )
			{
				const GateFictitiousVoxelMap* vmap=dynamic_cast<const GateFictitiousVoxelMap*> ( GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetFictitiousMap() );
				if ( vmap )
				{
					G4ThreeVector vs=vmap->GetGeometryVoxelReader()->GetVoxelSize();
					G4double lb=pTotalDiscreteProcess->GetTotalCrossSectionsTable()->GetEnergyLimitForGivenMaxCrossSection ( 4./sqrt ( vs[0]*vs[0]+vs[1]*vs[1]+vs[2]*vs[2] ) ); // estimates
					SetMinEnergy ( lb );
#ifdef G4VERBOSE
					G4cout << "GateFictitiousFastSimulationModel: Choose lower energy border = "<< lb/keV << " keV by using information about involved processes, materials, and voxel size." << G4endl;
#endif
				}
				else
				{
					G4cout << "Energy=" << GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetFictitiousEnergy() /MeV << " MeV" << G4endl;
					G4Exception ( "GateFictitiousFastSimulationModel::DoIt(const G4FastTrack&, G4FastStep&)", "Fictitious energy too small", FatalException,
					              "Cannot be negative or zero!" );
				}
			}
			else
			{
				SetMinEnergy ( GatePETVRTManager::GetInstance()->GetOrCreatePETVRTSettings()->GetFictitiousEnergy() );
			}
		}
		else
		{
			SetMinEnergy ( 0. );
		}




#ifdef G4VERBOSE
		if ( m_nDiscardEnergy>0 )
		{
			G4cout << "GateFictitiousFastSimulationModel: Use discard energy of "<< m_nDiscardEnergy / keV << " keV" << G4endl;

		}

		G4cout << "GateFictitiousFastSimulationModel: Use fictitious energy of "<< m_nMinEnergy / keV << " keV" << G4endl;
#endif

		if ( m_nMinEnergy>=m_nMaxEnergy )
		{
			G4Exception ( "GateFictitiousFastSimulationModel::DoIt(const G4FastTrack&, G4FastStep&)", "Fictitious eneryg too large", FatalException,
			              "Minimal energy larger than maximal energy! Use regularMatrix instead or lower manually the energy in '/gate/physics/gamma/setFictitiousEnergy' to a smaller positive value." );
		}

		pFictitiousMap->Check();
	}

	pCurrentFastTrack=&ft;
	// const_cast not nice but did not know how to do differently
	pCurrentTrack=const_cast<G4Track*> ( pCurrentFastTrack->GetPrimaryTrack() );

	m_nCurrentEnergy=pCurrentTrack->GetKineticEnergy();

	if ( m_nCurrentEnergy<=m_nDiscardEnergy )
	{
		DiscardPhoton();
		return;
	}


	pCurrentFastStep=&fs;

	m_nCurrentLocalDirection=ft.GetPrimaryTrackLocalDirection();
	m_nCurrentLocalPosition=ft.GetPrimaryTrackLocalPosition();

	m_nDistToOut=pEnvelopeSolid->DistanceToOut ( m_nCurrentLocalPosition,m_nCurrentLocalDirection ) +m_nSurfaceTolerance;

//G4cout << "E:" << pCurrentFastTrack->GetPrimaryTrack()->GetDynamicParticle()->GetKineticEnergy() << " " << G4endl;
	m_nCurrentInvFictCrossSection=1./pTotalDiscreteProcess->GetTotalCrossSectionsTable()->GetMaxCrossSection ( m_nCurrentEnergy );
	m_nTime=pCurrentTrack->GetGlobalTime();

	assert ( pTotalDiscreteProcess->GetNumberOfInteractionLengthLeft() <=0 );
	// const_cast not nice but did not know how to do differently
	pCurrentTrackStep=const_cast<G4Step*> ( pCurrentTrack->GetStep() );
	m_nNumSecondaries=0;
	m_nTotalPathLength=0.;
	m_nCurrentVelocity=pCurrentTrackStep->GetPreStepPoint()->GetVelocity();
	pTransform=pCurrentFastTrack->GetAffineTransformation();
	pInverseTransform=pCurrentFastTrack->GetInverseAffineTransformation();

	switch ( m_nApproximations )
	{
		case GatePETVRT::kVolumeTrace:

			VolumeTrace();
			break;
		case GatePETVRT::kDiscreteKleinNishina:
			assert ( false );
			break;
		default:
			G4Exception ( "GateFictitiousFastSimulationModel::DoIt(const G4FastTrack&, G4FastStep&)", "Approximations not known!", FatalException,
			              "This is a bug." );
			break;
	}
	AddSecondariesToFastStep();
	//cout << "return Secondaries" << endl;
}
void GateFictitiousFastSimulationModel::AttachPhantomSD ( GatePhantomSD* p )
{
	assert ( p!=NULL );
	pPhantomSD=p;
}


void GateFictitiousFastSimulationModel::VolumeTrace()
{
	G4ThreeVector finalPos;
	m_nPathLength=0;
	G4Material* currentMaterial;
	do
	{
		G4double fict;
		fict=-log ( G4UniformRand() ); // number mean free path lengths
		fict*=m_nCurrentInvFictCrossSection;      // distance including fictitious interaction
		m_nPathLength+=fict;   // add to total real distance
		if ( m_nPathLength>=m_nDistToOut ) // leaves Region before interaction would occur --> no interaction in envelope
		{
			Affine ( m_nCurrentLocalPosition,m_nCurrentLocalDirection,m_nDistToOut-m_nPathLength+fict );
			m_nTotalPathLength+=m_nDistToOut;
			m_nTime+=m_nDistToOut/m_nCurrentVelocity; //adjust time
			pCurrentFastStep->SetPrimaryTrackFinalTime ( m_nTime );
			pCurrentFastStep->ProposePrimaryTrackFinalPosition ( m_nCurrentLocalPosition,true );
			pCurrentFastStep->ProposePrimaryTrackFinalMomentumDirection ( m_nCurrentLocalDirection,true );
			pCurrentFastStep->SetPrimaryTrackFinalKineticEnergy ( m_nCurrentEnergy );
			pCurrentFastStep->SetPrimaryTrackPathLength ( m_nTotalPathLength ); //KEEP?
			return;
		}
		Affine ( m_nCurrentLocalPosition,m_nCurrentLocalDirection,fict ); // transport particle to new position
		currentMaterial=pFictitiousMap->GetMaterial ( m_nCurrentLocalPosition );
		assert ( pTotalCrossSectionsTable->GetCrossSection ( currentMaterial,m_nCurrentEnergy ) *m_nCurrentInvFictCrossSection<=1. );
	}
	while ( G4UniformRand() >=pTotalCrossSectionsTable->GetCrossSection ( currentMaterial,m_nCurrentEnergy ) *m_nCurrentInvFictCrossSection ); // check whether fictitious interaction

	// real interaction takes places:
	m_nTotalPathLength+=m_nPathLength; // update total path length
	m_nTime+=m_nPathLength/m_nCurrentVelocity; // update time

	G4StepPoint* preStepPoint=pCurrentTrackStep->GetPreStepPoint();
	preStepPoint->SetMaterial ( currentMaterial ); // give real material for G4 process


	// global position and direction
	G4ThreeVector globalPos=pInverseTransform->TransformPoint ( m_nCurrentLocalPosition );
	G4ThreeVector globalDir=pInverseTransform->TransformAxis ( m_nCurrentLocalDirection );


	// set position and time
	G4StepPoint*	postStepPoint=pCurrentTrackStep->GetPostStepPoint();
	postStepPoint->SetPosition ( globalPos );       // adjust position
	postStepPoint->SetGlobalTime ( m_nTime );   //adjust time

	// choose real G4 process
	postStepPoint->SetProcessDefinedStep ( pTotalDiscreteProcess->SampleActiveProcess ( currentMaterial,m_nCurrentEnergy ) );

	// change particle state and calculate secondaries if necessary using G4 methods
	G4VParticleChange* change=pTotalDiscreteProcess->PostStepDoIt ( *pCurrentTrack,*pCurrentTrackStep );

	// in order to process hit information
	if ( pPhantomSD!=NULL )
	{
		pCurrentTrackStep->SetStepLength ( m_nPathLength );
//		postStepPoint->SetPosition ( globalPos );       // adjust position
//		postStepPoint->SetGlobalTime ( m_nTime );   //adjust time
//		postStepPoint->SetProcessDefinedStep ( pTotalDiscreteProcess->GetActiveProcess() );

		pPhantomSD->ProcessHits ( pCurrentTrackStep,NULL );
	}

	// abort if G4 process says this
	if ( ( change->GetTrackStatus() == fStopAndKill ) )
	{
		pCurrentFastStep->KillPrimaryTrack();
		AddSecondaries ( change );
		//	change->Clear();
		return;
	}
	else if ( change->GetTrackStatus() ==fKillTrackAndSecondaries )
	{
		pCurrentFastStep->KillPrimaryTrack();
		for ( G4int i=0;i<m_nNumSecondaries;i++ )
			delete ( *m_pTrackFastVector ) [i];
		change->Clear();
		return;
	}

	G4ParticleChangeForGamma* change4gamma=dynamic_cast<G4ParticleChangeForGamma*> ( change );

	if ( change4gamma ) // G4VEmProcess
	{
		m_nCurrentEnergy=change4gamma->GetProposedKineticEnergy();
		pCurrentTrack->SetKineticEnergy ( m_nCurrentEnergy );
		if ( m_nCurrentEnergy<=m_nDiscardEnergy )
		{
			DiscardPhoton();
			AddSecondaries ( change );
			return;
		}
		if ( m_nCurrentEnergy<=m_nMinEnergy ) // energy too small --> standard tracking
		{
			pCurrentFastStep->SetPrimaryTrackFinalTime ( m_nTime );
			pCurrentFastStep->ProposePrimaryTrackFinalPosition ( m_nCurrentLocalPosition,true );
			pCurrentFastStep->ProposePrimaryTrackFinalMomentumDirection ( change4gamma->GetProposedMomentumDirection(),false );
			pCurrentFastStep->SetPrimaryTrackFinalKineticEnergy ( m_nCurrentEnergy );
			pCurrentFastStep->ProposePrimaryTrackFinalPolarization ( change4gamma->GetProposedPolarization(),false );
			pCurrentFastStep->SetPrimaryTrackPathLength ( m_nTotalPathLength );

			AddSecondaries ( change );
			return;
		}
		pCurrentTrack->SetPolarization ( change4gamma->GetProposedPolarization() );
		// update local momentum direction
		m_nCurrentLocalDirection=pTransform->TransformAxis ( change4gamma->GetProposedMomentumDirection() );
	}
	else // LowEnergy
	{
		G4ParticleChange* change4lowEnergy=dynamic_cast<G4ParticleChange*> ( change );
		if ( change4lowEnergy )
		{
			m_nCurrentEnergy=change4lowEnergy->GetEnergy();
			pCurrentTrack->SetKineticEnergy ( m_nCurrentEnergy );

			if ( m_nCurrentEnergy<=m_nDiscardEnergy )
			{
				DiscardPhoton();
				AddSecondaries ( change );
				return;
			}
			if ( m_nCurrentEnergy<=m_nMinEnergy ) // energy too small --> standard tracking
			{
				pCurrentFastStep->SetPrimaryTrackFinalTime ( m_nTime );
				pCurrentFastStep->ProposePrimaryTrackFinalPosition ( m_nCurrentLocalPosition,true );
				pCurrentFastStep->ProposePrimaryTrackFinalMomentumDirection ( *change4lowEnergy->GetMomentumDirection(),false );
				pCurrentFastStep->SetPrimaryTrackFinalKineticEnergy ( m_nCurrentEnergy );
				pCurrentFastStep->ProposePrimaryTrackFinalPolarization ( *change4lowEnergy->GetPolarization(),false );
				pCurrentFastStep->SetPrimaryTrackPathLength ( m_nTotalPathLength );
				AddSecondaries ( change );
				return;
			}
			pCurrentTrack->SetPolarization ( *change4lowEnergy->GetPolarization() );
			// update local momentum direction
			m_nCurrentLocalDirection=pTransform->TransformAxis ( * change4lowEnergy->GetMomentumDirection() );
		}
		else
		{
			G4Exception ( "GateFictitiousFastSimulationModel::VolumeTrace()", "ParticleChange type not recognized (wrong process)", FatalException,
			              "Aborting." );
		}
	}
	m_nCurrentVelocity=pCurrentTrackStep->GetPostStepPoint()->GetVelocity(); // update speed (in principle not necessary because photon)




	AddSecondaries ( change );

	// update dist to out
	m_nDistToOut=pEnvelopeSolid->DistanceToOut ( m_nCurrentLocalPosition,m_nCurrentLocalDirection ) +m_nSurfaceTolerance;

	// update inverse maximal cross section
	m_nCurrentInvFictCrossSection=1./pTotalDiscreteProcess->GetTotalCrossSectionsTable()->GetMaxCrossSection ( m_nCurrentEnergy );

	// continue tracking
	VolumeTrace();
}



void GateFictitiousFastSimulationModel::SetTotalDiscreteProcess ( GateTotalDiscreteProcess* p )
{
	assert ( p!=NULL );
	pTotalDiscreteProcess= p;
//	SetCrossSectionsTable ( pTotalDiscreteProcess->GetTotalCrossSectionsTable() );
}

void GateFictitiousFastSimulationModel::SetFictitiousMap ( GateVFictitiousMap* p )
{
	if ( ( pFictitiousMap!=NULL ) && ( p!=pFictitiousMap ) )
	{
		G4cout << "Warning! GateFictitiousFastSimulationModel::SetFictitiousVoxelMap: Fictitious voxel map pointer was already set. Fictitious voxel map pointer will be overwritten!" << G4endl;
	}
	pFictitiousMap=p;

	pEnvelopeSolid=pFictitiousMap->GetSolid();
}

bool GateFictitiousFastSimulationModel::SetParameters ( const GatePETVRTSettings* settings )
{
	if ( m_nInitialized )
	{
		G4Exception ( "GateFictitiousFastSimulationModel::SetParameters(const GatePETVRTSettings*)", "Already initialized", FatalException,
		              "Do nothing." );
		return true;
	}
	else
	{
		m_nApproximations=settings->GetApproximations();
		G4int counter=0;
		if ( settings->GetTotalDiscreteProcess() !=NULL )
		{
			SetTotalDiscreteProcess ( settings->GetTotalDiscreteProcess() );
			counter++;
		}
		if ( settings->GetFictitiousMap() !=NULL )
		{
			SetFictitiousMap ( settings->GetFictitiousMap() );
			counter++;
		}
		G4Envelope* anEnvelope=settings->GetEnvelope();
		if ( anEnvelope!=NULL )
		{
// Retrieves the Fast Simulation Manager or creates one if needed.
			G4FastSimulationManager* theFastSimulationManager;
			if ( ( theFastSimulationManager = anEnvelope->GetFastSimulationManager() ) == 0 )
				theFastSimulationManager = new G4FastSimulationManager ( anEnvelope,true );  //isunique
			// adds this model to the Fast Simulation Manager.
			theFastSimulationManager->AddFastSimulationModel ( this );
			assert ( theFastSimulationManager->GetEnvelope()->GetNumberOfRootVolumes() ==1 );
			assert ( anEnvelope->GetFastSimulationManager() );

			counter++;
		}
		GatePhantomSD* phantomSD=settings->GetPhantomSD();
		if ( phantomSD!=NULL )
		{
			pPhantomSD=phantomSD;
		}
		if ( counter==3 )
		{
			return true;
		}
		else
		{
			return false;
		}
	}
}
void GateFictitiousFastSimulationModel::SetMinEnergy ( G4double d )
{
	m_nMinEnergy=d;
	if ( m_nDiscardEnergy<=0 )
		if ( m_nMinEnergy<m_nAbsMinEnergy )
			m_nMinEnergy=m_nAbsMinEnergy;
}
void GateFictitiousFastSimulationModel::SetMaxEnergy ( G4double d )
{
	m_nMaxEnergy=d;
	if ( m_nMaxEnergy<m_nAbsMaxEnergy )
		m_nMaxEnergy=m_nAbsMaxEnergy;
}
