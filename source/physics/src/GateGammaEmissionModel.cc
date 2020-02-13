/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateGammaEmissionModel.hh"
#include <exception>
#include "G4LorentzVector.hh"
#include "Randomize.hh"
#include <GateMessageManager.hh>
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4ParticleTable.hh"


GateGammaEmissionModel::GateGammaEmissionModel() { pGammaDefinition = G4ParticleTable::GetParticleTable()->FindParticle( "gamma" ); }

GateGammaEmissionModel::~GateGammaEmissionModel() {}

G4int GateGammaEmissionModel::GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position)
{
 G4PrimaryVertex* vertex = new G4PrimaryVertex(particle_position, particle_time);
 G4PrimaryParticle* gamma = GetSingleGamma( fEmissionEnergy );
 gamma->SetUserInformation( GetPrimaryParticleInformation( gamma, GateEmittedGammaInformation::GammaKind::Single ) );
 vertex->SetPrimary( gamma );
 event->AddPrimaryVertex( vertex );

 return 1;
}

G4PrimaryParticle* GateGammaEmissionModel::GetSingleGamma( const G4double& energy ) const
{
 G4PrimaryParticle* gamma = new G4PrimaryParticle( pGammaDefinition );

 G4ThreeVector momentum_direction;

 if ( fUseFixedEmissionDirection ) { momentum_direction = fFixedEmissionDirection; }
 else { momentum_direction = GetUniformOnSphere(); }

 G4LorentzVector lv_gamma( momentum_direction.x(), momentum_direction.y(), momentum_direction.z(), 1.0 );
 lv_gamma *= energy;
 //Update gamma
 ///Momentum
 gamma->Set4Momentum( lv_gamma.px(), lv_gamma.py(), lv_gamma.pz(), lv_gamma.e() );
 ///Polarization
 gamma->SetPolarization( GetPolarization( gamma->GetMomentumDirection() ) );

 return gamma;
}
  
void GateGammaEmissionModel::SetFixedEmissionDirection( const G4ThreeVector& momentum_direction )
{
 if ( momentum_direction.mag() == 0 )
 { 
  NoticeError( G4String( __FUNCTION__ ), "fixed momemntum direction vector should be non zero vector" ); 
 }
 fFixedEmissionDirection = momentum_direction;
 fFixedEmissionDirection = fFixedEmissionDirection.unit();
 fUseFixedEmissionDirection = true;
}

void GateGammaEmissionModel::SetEnableFixedEmissionDirection( const G4bool enable ) { fUseFixedEmissionDirection = enable; }

void GateGammaEmissionModel::SetEmissionEnergy( const G4double& energy )
{
 if ( !( energy > 0.0 ) ) { NoticeError( G4String( __FUNCTION__ ), "emission energy should be positive value." ); }
 fEmissionEnergy = energy;
}

G4double GateGammaEmissionModel::GetEmissionEnergy() const { return fEmissionEnergy; }

void GateGammaEmissionModel::SetSeed( G4long seed )
{ 
 if ( seed < 0 ) { NoticeError( G4String( __FUNCTION__ ), "seed should be positive value." ); }
 G4Random::setTheSeed( seed ); 
}
 
G4long GateGammaEmissionModel::GetSeed() const { return G4Random::getTheSeed (); }

G4ThreeVector GateGammaEmissionModel::GetUniformOnSphere() const
{
 //Based on TRandom::Sphere
 G4double a = 0,b = 0, r2 = 1;
 while ( r2 > 0.25 ) 
 {
  a  = G4UniformRand() - 0.5;
  b  = G4UniformRand() - 0.5;
  r2 =  a*a + b*b;
 }
 
 G4double scale = 8.0 * sqrt(0.25 - r2);
 return G4ThreeVector( a * scale, b * scale, -1. + 8.0 * r2 );
}
  
G4ThreeVector GateGammaEmissionModel::GetPolarization( const G4ThreeVector& momentum ) const
{
 G4ThreeVector polarization(0.0,0.0,0.0);

 G4ThreeVector a0,b0,d0;
 d0 = momentum.unit();
 a0 = GetPerpendicularVector( d0 ).unit();
 b0 = d0.cross( a0 ).unit();
 G4double angle_radians = G4UniformRand() * M_PI;
 polarization = std::cos( angle_radians ) * a0 + std::sin( angle_radians ) * b0;
 polarization.unit();
 return polarization;
}
  
G4ThreeVector GateGammaEmissionModel::GetPerpendicularVector(const G4ThreeVector& v) const
{
 G4double dx = v.x();
 G4double dy = v.y();
 G4double dz = v.z();

 G4double x = dx < 0.0 ? -dx : dx;
 G4double y = dy < 0.0 ? -dy : dy;
 G4double z = dz < 0.0 ? -dz : dz;

 if (x < y) { return x < z ? G4ThreeVector(-dy,dx,0) : G4ThreeVector(0,-dz,dy); }
 else { return y < z ? G4ThreeVector(dz,0,-dx) : G4ThreeVector(-dy,dx,0); }
}
  
void GateGammaEmissionModel::SetModelName( const G4String model_name )
{
 if ( model_name.size() ==  0 ) { NoticeError( G4String( __FUNCTION__ ), "not provided model name." ); }
 fModelName = model_name;
}
  
void GateGammaEmissionModel::NoticeError( G4String method_name, G4String exception_description ) const
{
 G4String error_msg = fModelName + "::" + method_name + " : " + exception_description;
 GateError( error_msg );
}

GateEmittedGammaInformation* GateGammaEmissionModel::GetPrimaryParticleInformation( const G4PrimaryParticle* pp, const GateEmittedGammaInformation::GammaKind& gamma_kind ) const
{
 GateEmittedGammaInformation* egi = new GateEmittedGammaInformation();
 egi->SetSourceKind( GateEmittedGammaInformation::SourceKind::SingleGammaEmitter );
 egi->SetDecayModel( GateEmittedGammaInformation::DecayModel::None );
 egi->SetGammaKind( gamma_kind );
 egi->SetInitialPolarization( pp->GetPolarization() );
 return egi;
}
