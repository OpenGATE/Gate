/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <cmath>
#include <algorithm>

#include "Randomize.hh"
#include "G4DecayProducts.hh"
#include "G4LorentzVector.hh"
#include "G4ParticleTable.hh"
#include "G4DecayTable.hh"
#include "G4ParticleTable.hh"
#include "G4DecayTable.hh"
#include "G4ParticleDefinition.hh"

#include "GatePositroniumDecayModel.hh"

GatePositroniumDecayModel::Positronium::Positronium(const G4String name, G4double life_time, G4int annihilation_gammas_number ) : fName( name ), fLifeTime( life_time ), fAnnihilationGammasNumber( annihilation_gammas_number )
{
 G4ParticleDefinition* positronium_def = G4ParticleTable::GetParticleTable()->FindParticle( name );
 G4DecayTable* positronium_decay_table = positronium_def->GetDecayTable();
 pDecayChannel = positronium_decay_table->GetDecayChannel(0); 
}

void GatePositroniumDecayModel::Positronium::SetLifeTime( const G4double& life_time ) { fLifeTime = life_time; }

G4double GatePositroniumDecayModel::Positronium::GetLifeTime() const { return fLifeTime; }

G4String GatePositroniumDecayModel::Positronium::GetName() const { return fName; }

G4int GatePositroniumDecayModel::Positronium::GetAnnihilationGammasNumber() const { return fAnnihilationGammasNumber; }

G4DecayProducts* GatePositroniumDecayModel::Positronium::GetDecayProducts() { return pDecayChannel->DecayIt(); }

GatePositroniumDecayModel::GatePositroniumDecayModel() 
{
 SetModelName( "GatePositroniumDecayModel" );
}

GatePositroniumDecayModel::~GatePositroniumDecayModel() {}

void GatePositroniumDecayModel::SetPositroniumKind( GatePositroniumDecayModel::PositroniumKind positronium_kind ) { fPositroniumKind = positronium_kind; }

GatePositroniumDecayModel::PositroniumKind GatePositroniumDecayModel::GetPositroniumKind() const { return fPositroniumKind; }

void GatePositroniumDecayModel::SetDecayModel( GatePositroniumDecayModel::DecayModel decay_model ) { fDecayModel = decay_model; }

GatePositroniumDecayModel::DecayModel GatePositroniumDecayModel::GetDecayModel() const { return fDecayModel; }

void GatePositroniumDecayModel::SetPostroniumLifetime( G4String positronium_name, G4double life_time ) 
{ 
 if ( !( life_time > 0.0 ) ) { NoticeError( G4String( __FUNCTION__ ), "positronium life-time should be positive value." ); }
 
 if ( positronium_name == fParaPs.GetName() ) { fParaPs.SetLifeTime( life_time ); }
 else if ( positronium_name == fOrthoPs.GetName() ) { fOrthoPs.SetLifeTime( life_time ); }
 else { NoticeError( G4String( __FUNCTION__ ), "Unknown positronium name." ); }
}

void GatePositroniumDecayModel::SetPromptGammaEnergy( G4double prompt_energy )
{ 
 if ( !( prompt_energy > 0.0 ) ) { NoticeError( G4String( __FUNCTION__ ), "prompt gamma energy should be positive value." ); }
 fPromptGammaEnergy = prompt_energy;
}

G4double GatePositroniumDecayModel::GetPromptGammaEnergy() const { return fPromptGammaEnergy; }

void GatePositroniumDecayModel::SetParaPositroniumFraction( G4double fraction )
{
 fUsePositroniumFractions = true;
 fParaPositroniumFraction = fraction;
}

void GatePositroniumDecayModel::PreparePositroniumParametrization()
{
 if ( pInfoPs != nullptr )
 {
  if ( !fUsePositroniumFractions ) { return; }
  
  //Let's draw a positronium decay for current event
 
  if ( fParaPositroniumFraction >= G4UniformRand() ) { fPositroniumKind = PositroniumKind::pPs; }
  else { fPositroniumKind = PositroniumKind::oPs; }
 }

 switch ( fPositroniumKind ) 
 {
  case PositroniumKind::pPs:
   pInfoPs = &fParaPs;
   break;
  case PositroniumKind::oPs:
   pInfoPs = &fOrthoPs;
   break;
  default:
   NoticeError( G4String( __FUNCTION__ ), "improper chosen positronium kind." );
   break;
 };
}

GateEmittedGammaInformation* GatePositroniumDecayModel::GetPrimaryParticleInformation( const G4PrimaryParticle* pp, const GateEmittedGammaInformation::GammaKind& gamma_kind ) const
{
 GateEmittedGammaInformation* egi = new GateEmittedGammaInformation();

 GateEmittedGammaInformation::SourceKind source_kind = GateEmittedGammaInformation::SourceKind::ParaPositronium;
 GateEmittedGammaInformation::DecayModel decay_model = GateEmittedGammaInformation::DecayModel::Standard;

 if ( fPositroniumKind == PositroniumKind::oPs ) { source_kind = GateEmittedGammaInformation::SourceKind::OrthoPositronium; }
 if ( fDecayModel == DecayModel::WithPrompt ) { decay_model = GateEmittedGammaInformation::DecayModel::Deexcitation; }

 egi->SetSourceKind( source_kind );
 egi->SetDecayModel( decay_model );
 egi->SetGammaKind( gamma_kind );
 egi->SetInitialPolarization( pp->GetPolarization() );

 if ( gamma_kind == GateEmittedGammaInformation::GammaKind::Annihilation ){ egi->SetTimeShift( pInfoPs->GetLifeTime() ); }

 return egi;
}

G4int GatePositroniumDecayModel::GeneratePrimaryVertices(G4Event* event, G4double& particle_time, G4ThreeVector& particle_position )
{
 PreparePositroniumParametrization();
 G4int vertexes_number = 1;

 if ( fDecayModel == DecayModel::WithPrompt ) 
 { 
  ++vertexes_number;
  event->AddPrimaryVertex( GetPrimaryVertexFromDeexcitation(particle_time, particle_position) ); 
 }
 
 event->AddPrimaryVertex( GetPrimaryVertexFromPositroniumAnnihilation(particle_time, particle_position) );

 return vertexes_number;
}

G4PrimaryVertex* GatePositroniumDecayModel::GetPrimaryVertexFromDeexcitation(const G4double& particle_time, const  G4ThreeVector& particle_position )
{
 G4PrimaryVertex* vertex = new G4PrimaryVertex(particle_position, particle_time);
 vertex->SetPrimary( GetGammaFromDeexcitation() );
 return vertex;
}

G4PrimaryVertex* GatePositroniumDecayModel::GetPrimaryVertexFromPositroniumAnnihilation( const G4double& particle_time, const  G4ThreeVector& particle_position )
{
 G4double shifted_particle_time = particle_time + G4RandExponential::shoot( pInfoPs->GetLifeTime() );

 G4PrimaryVertex* vertex = new G4PrimaryVertex( particle_position, shifted_particle_time );
 std::vector<G4PrimaryParticle*> gammas = GetGammasFromPositroniumAnnihilation();
 std::for_each( gammas.begin(), gammas.end(), [&]( G4PrimaryParticle* gamma ) { vertex->SetPrimary( gamma ); } );
 return vertex;
}

G4PrimaryParticle* GatePositroniumDecayModel::GetGammaFromDeexcitation()
{
 G4PrimaryParticle* gamma = GetSingleGamma( fPromptGammaEnergy );
 gamma->SetUserInformation( GetPrimaryParticleInformation( gamma, GateEmittedGammaInformation::GammaKind::Prompt ) );
 return gamma;
}

std::vector<G4PrimaryParticle*> GatePositroniumDecayModel::GetGammasFromPositroniumAnnihilation()
{ 
 std::vector<G4PrimaryParticle*> gammas( pInfoPs->GetAnnihilationGammasNumber() ) ; 

 G4DecayProducts* decay_products = pInfoPs->GetDecayProducts();
 for ( G4int i = 0; i < pInfoPs->GetAnnihilationGammasNumber(); ++i )
 {
  G4PrimaryParticle* gamma = new G4PrimaryParticle( pGammaDefinition );

  G4DynamicParticle* dynamic_gamma = (*decay_products)[i];
  G4LorentzVector lv = dynamic_gamma->Get4Momentum();
  gamma->Set4Momentum( lv.px(), lv.py(), lv.pz(), lv.e() );
  gamma->SetPolarization( dynamic_gamma->GetPolarization() );
  gamma->SetUserInformation( GetPrimaryParticleInformation(  gamma, GateEmittedGammaInformation::GammaKind::Annihilation ) );
  gammas[i] = gamma;
 }
 delete decay_products;

 return gammas;
}

int MiniPositroniumDecayModel::getPositroniumDecayIndex(const std::vector<float>& fractions) {
  auto r = G4UniformRand(); 
  float curr_frac_cumulative = 0.0;
  for (int i = 0; i < fractions.size(); ++i) {
    curr_frac_cumulative = curr_frac_cumulative + fractions[i];
    if(r<= curr_frac_cumulative) return i;   
 }
  return -1;
}

G4PrimaryVertex* MiniPositroniumDecayModel::GetPrimaryVertexFromPositroniumAnnihilation(G4double particle_time, const G4ThreeVector& particle_position, int decayIndex)
{

 G4double shifted_particle_time = particle_time + G4RandExponential::shoot(fModelParams.fLifetimes[decayIndex]);

 G4PrimaryVertex* vertex = new G4PrimaryVertex( particle_position, shifted_particle_time );
 std::vector<G4PrimaryParticle*> gammas = GetGammasFromPositroniumAnnihilation(decayIndex);
 std::for_each( gammas.begin(), gammas.end(), [&]( G4PrimaryParticle* gamma ) { vertex->SetPrimary( gamma ); } );
 return vertex;
}

// For this moment without prompt
G4int MiniPositroniumDecayModel::GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position)
{
  auto decayIndex = MiniPositroniumDecayModel::getPositroniumDecayIndex(fModelParams.fFractions);
  G4int number_of_vertices = 1;
 if(fModelParams.fIsPromptPhoton[decayIndex]) 
 { 
  ++number_of_vertices ;
  event->AddPrimaryVertex(GetPrimaryVertexFromDeexcitation(particle_time, particle_position, decayIndex)); 
 }
  event->AddPrimaryVertex(GetPrimaryVertexFromPositroniumAnnihilation(particle_time, particle_position, decayIndex));
  return number_of_vertices;
} 

G4PrimaryParticle* MiniPositroniumDecayModel::GetGammaFromDeexcitation(int decayIndex)
{
 G4PrimaryParticle* gamma = GetSingleGamma(fModelParams.fPromptPhotonEnergy[decayIndex]);
 gamma->SetUserInformation( GetPrimaryParticleInformation( gamma, GateEmittedGammaInformation::GammaKind::Prompt ) );
 return gamma;
}

G4PrimaryVertex* MiniPositroniumDecayModel::GetPrimaryVertexFromDeexcitation(G4double particle_time, const  G4ThreeVector& particle_position, int decayIndex)
{
 G4PrimaryVertex* vertex = new G4PrimaryVertex(particle_position, particle_time);
 vertex->SetPrimary(GetGammaFromDeexcitation(decayIndex));
 return vertex;
}

std::vector<G4PrimaryParticle*> MiniPositroniumDecayModel::GetGammasFromPositroniumAnnihilation(int decayIndex)
{ 
 int annihilation_gammas_number = fPositroniumDecayChannel[decayIndex].GetAnnihilationGammasNumber();
 std::vector<G4PrimaryParticle*> gammas(annihilation_gammas_number); 

 G4DecayProducts* decay_products = fPositroniumDecayChannel[decayIndex].GetDecayProducts();
 for ( G4int i = 0; i < annihilation_gammas_number; ++i )
 {
  G4PrimaryParticle* gamma = new G4PrimaryParticle( pGammaDefinition );

  G4DynamicParticle* dynamic_gamma = (*decay_products)[i];
  G4LorentzVector lv = dynamic_gamma->Get4Momentum();
  gamma->Set4Momentum( lv.px(), lv.py(), lv.pz(), lv.e() );
  gamma->SetPolarization( dynamic_gamma->GetPolarization() );
  gamma->SetUserInformation( GetPrimaryParticleInformation(  gamma, GateEmittedGammaInformation::GammaKind::Annihilation ) );
  gammas[i] = gamma;
 }
 delete decay_products;

 return gammas;
}

