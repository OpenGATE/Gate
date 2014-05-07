/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateCrossSectionsTable.hh"
#include "G4Material.hh"
#include <cassert>
#include "G4ParticleDefinition.hh"
#include "G4PhysicsLinearVector.hh"
#include "G4EmCalculator.hh"
#include "G4ProductionCutsTable.hh"
#include "G4MaterialTable.hh"
#include "G4ios.hh"
#include "GatePETVRTManager.hh"
#include "GateMaterialTableToProductionCutsTable.hh"
#include "G4LivermorePolarizedRayleighModel.hh"
#include "G4LivermoreRayleighModel.hh"
#include "G4VEmProcess.hh"
#include "G4ForceCondition.hh"
#include "G4LivermoreComptonModel.hh"
#include "G4LivermorePhotoElectricModel.hh"


using namespace std;
#define SAFETYFACTOR 1.0001

G4int GateCrossSectionsTable::PARTICLE_NAME_LENGTH=20;

//G4EmCalculator GateCrossSectionsTable::m_sEmCalculator;

GateCrossSectionsTable::GateCrossSectionsTable ( G4double minEnergy, G4double maxEnergy,  G4int physicsVectorBinNumber, const G4ParticleDefinition* pdef, const vector<G4VDiscreteProcess*>& processes ) :G4PhysicsTable(),m_oInvDensity(),m_oMaterialVec(), m_oProcessVec ( processes ),m_pMaxCrossSection ( NULL )
{
	assert ( pdef == G4Gamma::GammaDefinition() ); // perhaps it works for other particles, perhaps not... did not think about that
	m_nMinEnergy=minEnergy;
	m_nMaxEnergy=maxEnergy;
	assert ( m_nMaxEnergy>m_nMinEnergy );

	m_nPhysicsVectorBinNumber=physicsVectorBinNumber;
	pParticleDefinition=pdef;

	GatePETVRTManager* man=GatePETVRTManager::GetInstance();
	pMaterialTableToProductionCutsTable=man->GetMaterialTableToProductionCutsTable();
	m_nVerbose=3;

}

GateCrossSectionsTable::GateCrossSectionsTable ( G4double minEnergy, G4double maxEnergy,  G4int physicsVectorBinNumber, const G4ParticleDefinition* pdef, G4VDiscreteProcess& process ) :G4PhysicsTable(),m_oInvDensity(),m_oMaterialVec(), m_oProcessVec ( 1, &process ),m_pMaxCrossSection ( NULL )
{

	assert ( pdef == G4Gamma::GammaDefinition() ); // perhaps it works for other particles, perhaps not... did not think about that
	m_nMinEnergy=minEnergy;
	m_nMaxEnergy=maxEnergy;
	assert ( m_nMaxEnergy>m_nMinEnergy );


	m_nPhysicsVectorBinNumber=physicsVectorBinNumber;
	pParticleDefinition=pdef;

	GatePETVRTManager* man=GatePETVRTManager::GetInstance();
	pMaterialTableToProductionCutsTable=man->GetMaterialTableToProductionCutsTable();

}

GateCrossSectionsTable::GateCrossSectionsTable ( ifstream& in, bool ascii, const vector<G4VDiscreteProcess*>& processes ) :G4PhysicsTable(),m_oInvDensity(),m_oMaterialVec(), m_oProcessVec ( processes ),m_pMaxCrossSection ( NULL )
{
	GatePETVRTManager* man=GatePETVRTManager::GetInstance();
	pMaterialTableToProductionCutsTable=man->GetMaterialTableToProductionCutsTable();


	RetrieveTable ( in,ascii );
}


GateCrossSectionsTable::~GateCrossSectionsTable()
{
	if ( m_pMaxCrossSection!=NULL ) delete m_pMaxCrossSection;
	clearAndDestroy();
}

void GateCrossSectionsTable::SetVerbose(int v)
{
	m_nVerbose=v;
}

size_t GateCrossSectionsTable::AddMaterial ( const G4MaterialCutsCouple* couple ) // returns index for that material
{

        static G4EmCalculator m_sEmCalculator;

	if ( m_pMaxCrossSection!=NULL )
	{
		G4cout << "GateCrossSectionsTable::AddMaterial( const G4Material* mat ) : Added material AFTER building maximal! This will most probably lead to wrong results!" << G4endl;
	}

	G4PhysicsLinearVector* a=new G4PhysicsLinearVector ( m_nMinEnergy, m_nMaxEnergy, m_nPhysicsVectorBinNumber );
	const G4double delta= ( m_nMaxEnergy-m_nMinEnergy ) /m_nPhysicsVectorBinNumber;
	G4DynamicParticle* partTmp =new G4DynamicParticle ( const_cast<G4ParticleDefinition*> ( pParticleDefinition ),G4ThreeVector ( 0.,0.,0. ) ); // deleted by trackTmp
	G4StepPoint* point=new G4StepPoint(); // deleted by step
	G4Track trackTmp ( partTmp,0,G4ThreeVector ( 0.,0.,0. ) );
	G4Step step;
	step.SetPreStepPoint ( point );
	trackTmp.SetStep ( &step );
	G4ProductionCuts product;
	if (m_nVerbose>=3)
	{
		G4cout << "GateCrossSectionsTable::AddMaterial( const G4Material* mat ) : add " << couple->GetMaterial()->GetName() << G4endl;
	}
	for ( size_t i=0;i<a->GetVectorLength();i++ )
	{
		G4double energy=m_nMinEnergy+delta*i;
		G4double b=0;
		G4double c=0.;
		for ( size_t j=0;j<m_oProcessVec.size();j++ )
		{
			if ( dynamic_cast<G4VEmProcess*> ( m_oProcessVec[j] ) )
			{
				c=m_sEmCalculator.ComputeCrossSectionPerVolume ( energy, pParticleDefinition, m_oProcessVec[j]->GetProcessName(), couple->GetMaterial() );
				assert ( c>0 );
				b+=c;
			}
			else
			{
				trackTmp.SetKineticEnergy ( energy );
				G4StepPoint* point=const_cast<G4StepPoint*> ( trackTmp.GetStep()->GetPreStepPoint() );
				point->SetMaterialCutsCouple ( couple );
				//G4ForceCondition forc=NotForced;
				G4LivermoreRayleighModel* ray= dynamic_cast<G4LivermoreRayleighModel*> ( m_oProcessVec[j] );
				if ( ray )
				{
					//c=ray->DumpMeanFreePath ( trackTmp,0., &forc );
          c=ray->ComputeMeanFreePath(pParticleDefinition, energy, couple->GetMaterial());
					assert ( c>0 );
					b+=1./c;
				}
				else
				{
					G4LivermorePolarizedRayleighModel* polray=dynamic_cast<G4LivermorePolarizedRayleighModel*> ( m_oProcessVec[j] );
					if ( polray )

					{
						//c=polray->DumpMeanFreePath ( trackTmp,0.,&forc );
						c=polray->ComputeMeanFreePath(pParticleDefinition, energy, couple->GetMaterial());						
						assert ( c>0 );
						b+=1./c;
					}
					else
					{
						G4LivermoreComptonModel*
						lcomp=dynamic_cast<G4LivermoreComptonModel*> ( m_oProcessVec[j] );
						if ( lcomp )

						{
							//c=lcomp->DumpMeanFreePath ( trackTmp,0.,&forc );
              c=lcomp->ComputeMeanFreePath(pParticleDefinition, energy, couple->GetMaterial());
							assert ( c>0 );
							b+=1./c;
						}
						else
						{
							/*  Does not work for some reason
														G4LivermorePhotoElectricModel*
														lphot=dynamic_cast<G4LivermorePhotoElectricModel*> ( m_oProcessVec[j] );
														if ( lphot )

														{
															//c=lphot->DumpMeanFreePath ( trackTmp,0.,&forc );
                              c=lphot->ComputeMeanFreePath(pParticleDefinition, energy, couple->GetMaterial());
															assert ( c>0 );
															b+=1./c;
														}
														else
														{
							*/
							G4Exception ( "GateCrossSectionsTable::AddMaterial(const G4MaterialCutsCouple*)", G4String ( "'" + m_oProcessVec[j]->GetProcessName() +"' is neither G4VEmProcess (='standard') nor G4LivermoreRayleighModel, G4LivermorePolarizedRayleighModel, G4LivermoreComptonModel. At present other processes cannot be used together with fictitious." ).c_str(), FatalException,"Cannot build fast tables." );

							//						}
						}
					}
				}
			}
		}
		a->PutValue ( i,b );
	}
	push_back ( a );
	m_oInvDensity.push_back ( 1./couple->GetMaterial()->GetDensity() );
	m_oMaterialVec.push_back ( couple->GetMaterial() );
	/*
		G4String totalname="table_"+couple->GetMaterial()->GetName();

			for ( size_t j=0;j<m_oProcessVec.size();j++ )
			totalname+=m_oProcessVec[j]->GetProcessName();
		ofstream fout;
		G4cout<< totalname.c_str() << G4endl;
		fout.open(totalname.c_str());
		a->Store(fout,true);
		fout.close();
	*/
	return ( size()-1 );
}



size_t GateCrossSectionsTable::SetAndBuildProductionMaterialTable () // returns index for that material
{

	if ( size() >0 )
		G4cout << "GateCrossSectionsTable::SetAndBuildProductionMaterialTable (): Delete old materials in table!" << G4endl;
	const G4ProductionCutsTable* table=G4ProductionCutsTable::GetProductionCutsTable ();
	size_t	nMaterials = table->GetTableSize ();
//	size_t nTotalMaterials=G4Material::GetNumberOfMaterials();


	pMaterialTableToProductionCutsTable->Update();
	for ( size_t m=0; m<nMaterials; m++ )
	{
		const G4MaterialCutsCouple* couple=table->GetMaterialCutsCouple ( m );
		assert ( pMaterialTableToProductionCutsTable->P2M ( m ) ==static_cast<G4int> ( couple->GetMaterial()->GetIndex() ) );
		//const G4Material* mat= couple->GetMaterial();
		AddMaterial ( couple ); // add material and build table
	}
	CheckInternalProductionMaterialTable(); // can be removed later, just to be sure
	return ( size()-1 );
}

G4int GateCrossSectionsTable::GetIndex ( const G4Material* mat ) const
{
	return pMaterialTableToProductionCutsTable->M2P ( mat->GetIndex() );
}



bool GateCrossSectionsTable::CheckInternalProductionMaterialTable() const // should return true if internal table correct
{
	const G4ProductionCutsTable* table=G4ProductionCutsTable::GetProductionCutsTable ();
	size_t	nMaterials = table->GetTableSize ();
	if ( nMaterials!=m_oMaterialVec.size() ) return false;
	size_t nTotalMaterials=G4Material::GetNumberOfMaterials();
	for ( size_t m=0; m<nTotalMaterials; m++ )
	{
		G4int n=pMaterialTableToProductionCutsTable->M2P ( m );
		if ( n>=0 ) // also in production
		{
			if ( m_oMaterialVec[n]->GetIndex() !=m ) return false;
		}
	}
	return true;
}


bool GateCrossSectionsTable::BuildMaxCrossSection ( const vector<G4Material*> & vec )
{
	if ( vec.size() <=0 )
	{
		G4Exception ( "BuildMaxCrossSection(const vector<G4Material*>&)", "Vector empty!", FatalException,"No Materials in vector." );
		return false;
	}
	size_t i=0;
	vector<size_t> involved_mat_index;

	for ( i=0;i<m_oMaterialVec.size();i++ )
	{
		for ( size_t j=0;j<vec.size();j++ )
		{
			if ( m_oMaterialVec[i]==vec[j] )
			{
				involved_mat_index.push_back ( i );
#ifdef G4VERBOSE
		G4cout << "BuildMaxCrossSection for phantom: Add material "<< m_oMaterialVec[i]->GetName() << G4endl;
#endif
				break;
			}
		}
	}
	if ( involved_mat_index.size() <=0 )
	{
		G4Exception ( "BuildMaxCrossSection(const vector<G4Material*>&)", "Materials not found in table", FatalException,"Aborting." );
		return false;
	}



	if ( m_pMaxCrossSection!=NULL ) delete m_pMaxCrossSection;
	m_pMaxCrossSection=new G4PhysicsLinearVector ( m_nMinEnergy, m_nMaxEnergy, m_nPhysicsVectorBinNumber );
//	const G4double delta= ( m_nMaxEnergy-m_nMinEnergy ) /m_nPhysicsVectorBinNumber;
	for ( size_t i=0;i<m_pMaxCrossSection->GetVectorLength();i++ )
	{
//		G4double energy=m_nMinEnergy+delta*i;
		G4double b=-1e9;
		for ( size_t j=0;j<involved_mat_index.size();j++ )
		{

			if ( operator[] ( involved_mat_index[j] )->operator[] ( i ) >b ) b=operator[] ( involved_mat_index[j] )->operator[] ( i );
		}
		if (b<0)
		{
				G4Exception ( "BuildMaxCrossSection(const vector<G4Material*>&)", "Negative maximal cross section!", FatalException,"Aborting." );

		}
		m_pMaxCrossSection->PutValue ( i,b*SAFETYFACTOR);
	}
	return true;
}

void GateCrossSectionsTable::Store ( std::ofstream& out, bool ascii, size_t num ) const
{
	assert ( num<length() );
	operator() ( num )->Store ( out,ascii );
	if ( ascii )
		out << m_oInvDensity[num] << endl;
	else
		out.write ( reinterpret_cast<const char*> ( &m_oInvDensity[num] ),sizeof ( m_oInvDensity[num] ) );
}

void GateCrossSectionsTable::Retrieve ( std::ifstream& in, bool ascii, size_t num )
{
	assert ( num<length() );
	operator() ( num )->Retrieve ( in,ascii );
	if ( ascii )
		in >> m_oInvDensity[num];
	else
		in.read ( reinterpret_cast<char*> ( &m_oInvDensity[num] ),sizeof ( m_oInvDensity[num] ) );
}

void GateCrossSectionsTable::StoreTable ( std::ofstream& out, bool ascii ) const
{
	size_t l=length();
	if ( ascii )
	{
		out << pParticleDefinition->GetParticleName() << endl;
		out << m_nMinEnergy << " " << m_nMaxEnergy << endl;
		out << m_nPhysicsVectorBinNumber << endl;
		out << length() << endl;
	}
	else
	{
		out.write ( reinterpret_cast<const char*> ( &pParticleDefinition->GetParticleName() ), PARTICLE_NAME_LENGTH );
		out.write ( reinterpret_cast<const char*> ( &m_nMinEnergy ),sizeof ( m_nMinEnergy ) );
		out.write ( reinterpret_cast<const char*> ( &m_nMaxEnergy ),sizeof ( m_nMaxEnergy ) );
		out.write ( reinterpret_cast<const char*> ( &m_nPhysicsVectorBinNumber ),sizeof ( m_nPhysicsVectorBinNumber ) );
		out.write ( reinterpret_cast<const char*> ( &l ),sizeof ( l ) );

	}
	for ( size_t i=0;i<l; i++ )
	{
		Store ( out,ascii,i );
	}
}

void GateCrossSectionsTable::RetrieveTable ( std::ifstream& in, bool ascii )
{
	size_t tmpsize=0;
	pParticleDefinition=G4Gamma::GammaDefinition();
	if ( ascii )
	{
		std::string name;
		in >> name;
		if ( name!=pParticleDefinition->GetParticleName() )
		{
			G4cout << "Try to Retrieve CrossSectionsTable for non-gamma! Particle panic!" << G4endl;
			assert ( name==pParticleDefinition->GetParticleName() );
			exit ( EXIT_FAILURE );
		}

		in >> m_nMinEnergy;
		in >> m_nMaxEnergy;
		in >> m_nPhysicsVectorBinNumber;
		in >> tmpsize;

	}
	else
	{
		std::string name ( PARTICLE_NAME_LENGTH,'\0' );

		in.read ( reinterpret_cast<char*> ( &name ), PARTICLE_NAME_LENGTH );
		if ( name!=pParticleDefinition->GetParticleName() )
		{
			G4cout << "Try to Retrieve CrossSectionsTable for non-gamma! Particle panic!" << G4endl;
			assert ( name==pParticleDefinition->GetParticleName() );
			exit ( EXIT_FAILURE );
		}

		in.read ( reinterpret_cast<char*> ( &m_nMinEnergy ),sizeof ( m_nMinEnergy ) );
		in.read ( reinterpret_cast<char*> ( &m_nMaxEnergy ),sizeof ( m_nMaxEnergy ) );
		in.read ( reinterpret_cast<char*> ( &m_nPhysicsVectorBinNumber ),sizeof ( m_nPhysicsVectorBinNumber ) );
		in.read ( reinterpret_cast<char*> ( &tmpsize ) ,sizeof ( tmpsize ) );
	}
	clearAndDestroy();
	m_oInvDensity.resize ( tmpsize );

	for ( size_t i=0;i<tmpsize; i++ )
	{
		G4cout << "Read cross section table for fictitious material no " << i << G4endl;
		assert ( !in.fail() );
		G4PhysicsLinearVector* dummy=new G4PhysicsLinearVector ( m_nMinEnergy,m_nMaxEnergy ,m_nPhysicsVectorBinNumber );
		push_back ( dummy );
		Retrieve ( in,ascii,length()-1 );
	}
	assert (	tmpsize==length() );
	assert ( m_nMaxEnergy>m_nMinEnergy );

}

G4double GateCrossSectionsTable::GetEnergyLimitForGivenMaxCrossSection ( G4double crossSection ) const
{
	G4int i=0;
	for ( i=m_pMaxCrossSection->GetVectorLength();i>=0;--i )
	{
		if ( ( *m_pMaxCrossSection ) [i]>=crossSection ) break;
	}
	const G4double delta= ( m_nMaxEnergy-m_nMinEnergy ) /m_nPhysicsVectorBinNumber;
	return 		m_nMinEnergy+delta*i;
}
