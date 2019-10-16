/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateExtendedVSource.hh"
#include "G4Event.hh"
#include "GateExtendedVSourceManager.hh"
#include "G4PrimaryParticle.hh"
#include "TMath.h"

GateExtendedVSource::GateExtendedVSource( G4String name ) : GateVSource( name )
{
 GateGammaSourceModels::InitModels();

 //This call only at the end of constructor
 pSourceMessenger = new GateExtendedVSourceMessenger( this );
}

GateExtendedVSource::~GateExtendedVSource()
{
 delete pSourceMessenger;
}

G4int GateExtendedVSource::GeneratePrimaries( G4Event* event )
{
 if ( ptrGammaSourceModel ) 
 {
  GateVSource::SetParticleTime( GetTime() );
  //Then we set number of particles which will be generated
  GateVSource::SetNumberOfParticles( ptrGammaSourceModel->GetParticlesNumber() );
  //First we attach our event volume
  GateVSource::GeneratePrimaryVertex( event );

  std::vector<G4PrimaryParticle*> particles;
  particles.resize( ptrGammaSourceModel->GetParticlesNumber() );

  for ( int particleIndex = 0; particleIndex<ptrGammaSourceModel->GetParticlesNumber(); ++particleIndex ) { particles[ particleIndex ] = event->GetPrimaryVertex( 0 )->GetPrimary( particleIndex ); }

  // And finally we generate particles
  ptrGammaSourceModel->GetGammaParticles( particles );
 } 
 else 
 {
  if ( InitModel() ) { return GeneratePrimaries( event ); }
  else 
  {
   G4String commands = GateExtendedVSourceManager::GetInstance()->GetGammaSourceModelsNames();

   if ( commands.size() > 0 ) { GateError( "Sorry, I don't know the source type '" << GetType() << "'. Known source types are "<< commands ); }
   else { GateError( "Sorry, I don't know the source type '"<< GetType() << "'. There are no definided types" ); }
  }
 }

 return 1;
}

bool GateExtendedVSource::InitModel()
{
 ptrGammaSourceModel = GateExtendedVSourceManager::GetInstance()->GetGammaSourceModelByName( GetType() );
 
 if( ptrGammaSourceModel != nullptr )
 {
  ptrGammaSourceModel->SetLinearPolarizationAngle( fLinearPolarizationAngle, false );
  ptrGammaSourceModel->SetUnpolarizedGammaGeneration( fUseUnpolarizedParticles );
  ptrGammaSourceModel->SetSeedForRandomGenerator( fSeedForRandomGenerator );
  ptrGammaSourceModel->SetPromptGammaEnergy( fPromptGammaEnergy );
  return true;
 }
 return false;
} 

void GateExtendedVSource::SetSeedForRandomGenerator( const unsigned int seed ) {  fSeedForRandomGenerator = seed; }

void GateExtendedVSource::SetPromptGammaEnergy( const double energy ) { fPromptGammaEnergy = energy; }

void GateExtendedVSource::SetLinearPolarizationAngle( const double angle ){ fLinearPolarizationAngle = TMath::DegToRad() * angle; }

void GateExtendedVSource::SetUnpolarizedParticlesGenerating( const bool use_unpolarized ){ fUseUnpolarizedParticles = use_unpolarized; }

