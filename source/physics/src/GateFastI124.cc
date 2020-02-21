/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateFastI124.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "GateSourceMgr.hh"

GateFastI124::GateFastI124( GateVSource* source )
{
	m_simpleDecay = 0;
	m_particleVector = 0;
	m_source = source;
}

GateFastI124::~GateFastI124()
{
//	vector<psd>::iterator i = this->m_particleVector->begin();
//	for( ; i != this->m_particleVector->end(); ++i ) {
//		delete i;
//	}
	this->m_particleVector->clear();
	delete this->m_particleVector;
	this->m_particleVector = 0;
	
	
	delete m_simpleDecay;
	
}
void GateFastI124::InitializeFastI124()
{	
// Forces fixed energy.  Real energy will be set later
	m_source->SetNumberOfParticles( 1 );
	m_source->GetEneDist()->SetEnergyDisType( "Mono" );
	m_source->SetParticleTime( m_source->GetTime() );

	
	// This defines the 13 transitions forming the simplified scheme
  // Entries are:
  //             for gammas :      current state / next state / cumulative probability / particle emitted (gamma)    / energy
  //             for e+     :      current state / next state / cumulative probability / particle emitted (e+)       / max energy ...
  //                               ... / amplitude of majoring function / normalisation factor for energy distribution (Fermi function) / atomic number

	m_simpleDecay = new GateSimplifiedDecay();
	m_particleVector = new std::vector<psd>;
	
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 1, 0.0175,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     1.376   )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 1, 0.0488,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     1.509   )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 1, 0.1189,  mem_fn( &GateSimplifiedDecayTransition::issueNone )              )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 2, 0.2359,  mem_fn( &GateSimplifiedDecayTransition::issuePositron),  1.534,  1.4,  0.4407471595713562, 53) );
  
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 2, 0.3447,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     1.691   )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 2, 0.3515,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     2.283   )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 2, 0.3598,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     0.645   )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0, 2, 0.6422,  mem_fn( &GateSimplifiedDecayTransition::issueNone )              )  );
  
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0,-1, 0.7502,  mem_fn( &GateSimplifiedDecayTransition::issuePositron),  2.137,  1.0,  0.10072654633851122,53) );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(0,-1, 1.,      mem_fn( &GateSimplifiedDecayTransition::issueNone )              )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(1, 2, 0.8690,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     0.722   )  );
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(1,-1, 1.0000,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     1.325   )  );
  
  m_simpleDecay->addTransition( new GateSimplifiedDecayTransition(2,-1, 1.0000,  mem_fn( &GateSimplifiedDecayTransition::issueGamma),     0.602   )  );
}

void GateFastI124::GenerateVertex( G4Event* aEvent )
{
	// Forces fixed energy.  Real energy will be set later
	//m_source->SetNumberOfParticles( 1 );
	//m_source->GetEneDist()->SetEnergyDisType( "Mono" );
	
	// Generate the vector of pairs particle name - energy
	m_particleVector->clear();
	m_simpleDecay->doDecay( m_particleVector );

	// If there are particles to generate
	if( !m_particleVector->empty() )
	{
		// create a new vertex at a new position/time
		m_source->GetPosDist()->GenerateOne();
		G4PrimaryVertex* vertex = new G4PrimaryVertex(
			m_source->GetParticlePosition(), m_source->GetTime() );
		
		// From the vector, create particles with own direction, type and energy
		for( std::vector<psd>::iterator it = m_particleVector->begin(); 
				 it != m_particleVector->end(); ++it )
		{
			if( m_source->GetVerboseLevel() > 1 ) 
					 G4cout << "GateVSource::GeneratePrimaries - fastI124 " << (*it).first
				 					<< ' ' << (*it).second << Gateendl;
									
			m_source->SetParticleDefinition( 
				G4ParticleTable::GetParticleTable()->FindParticle( (*it).first )  );
			
			m_source->GetEneDist()->SetMonoEnergy( (*it).second );
			m_source->GetEneDist()->GenerateOne( m_source->GetParticleDefinition() );
			m_source->GetAngDist()->GenerateOne();
			m_source->GeneratePrimaryVertex( aEvent );
		}
		aEvent->AddPrimaryVertex( vertex );
	}
	else
	{
		// 0 generated particles is not accept by the software. Generate a vertex
		// until the number of particle at least 1. And increment the time if 0
		G4double time = (GateSourceMgr::GetInstance())->GetTime();
		G4double nextTime = m_source->GetNextTime( time );
		(GateSourceMgr::GetInstance())->SetTime( time + nextTime );
		m_source->SetTime( time + nextTime );
		GenerateVertex( aEvent );
	}
}


