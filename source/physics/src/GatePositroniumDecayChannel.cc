/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GatePositroniumDecayChannel.hh"
#include "G4DynamicParticle.hh"
#include "G4DecayProducts.hh"
#include "Randomize.hh"
#include "G4LorentzVector.hh"

GatePositroniumDecayChannel::GatePositroniumDecayChannel(const G4String& theParentName, G4double theBR)
{
 G4int daughters_number = 0;
 
 if ( theParentName == kParaPositroniumName ) 
 { 
  daughters_number = kParaPositroniumAnnihilationGammasNumber;
  fPositroniumKind = GatePositroniumDecayChannel::PositroniumKind::ParaPositronium;
 }
 else if ( theParentName == kOrthoPositroniumName )
 { 
  daughters_number = kOrthoPositroniumAnnihilationGammasNumber;
  fPositroniumKind = GatePositroniumDecayChannel::PositroniumKind::OrthoPositronium;
 }
 else
 {
  #ifdef G4VERBOSE
  if (GetVerboseLevel()>0)
  {
   G4cout << "GatePositroniumDecayChannel:: constructor :";
   G4cout << " parent particle is not positronium (pPs,oPs) but ";
   G4cout << theParentName << G4endl;
  }
  #endif
 }

 SetParentMass( kPositroniumMass );
 SetBR( theBR );
 SetParent( theParentName );
 SetNumberOfDaughters( daughters_number );
 for ( G4int daughter_index = 0; daughter_index < daughters_number; ++daughter_index ) { SetDaughter( daughter_index, kDaughterName ); }
}

GatePositroniumDecayChannel::~GatePositroniumDecayChannel() {}

G4DecayProducts* GatePositroniumDecayChannel::DecayIt(G4double)
{
 switch ( fPositroniumKind )
 {
  case GatePositroniumDecayChannel::PositroniumKind::ParaPositronium:
   return DecayParaPositronium();
  case GatePositroniumDecayChannel::PositroniumKind::OrthoPositronium:
   return DecayOrthoPositronium();
  default:
   G4cout << "GatePositroniumDecayChannel::DecayIt ";
   G4cout << *parent_name << " can not decay " << G4endl;
   DumpInfo();
   return nullptr;
 };
}

G4DecayProducts* GatePositroniumDecayChannel::DecayParaPositronium()
{
 G4DecayProducts* decay_products = G4GeneralPhaseSpaceDecay::DecayIt();

 G4DynamicParticle* gamma_1 = (*decay_products)[0];
 G4DynamicParticle* gamma_2 = (*decay_products)[1];

 ///Polarization
 G4ThreeVector polarization_gamma_1 = GetPolarization( gamma_1->GetMomentumDirection() );
 G4ThreeVector polarization_gamma_2 = gamma_1->GetMomentumDirection().cross( polarization_gamma_1 );

 gamma_1->SetPolarization( polarization_gamma_1.x(), polarization_gamma_1.y(), polarization_gamma_1.z() );
 gamma_2->SetPolarization( polarization_gamma_2.x(), polarization_gamma_2.y(), polarization_gamma_2.z() );

 return decay_products;
}

G4DecayProducts* GatePositroniumDecayChannel::DecayOrthoPositronium()
{
 G4LorentzVector lv_gamma_1, lv_gamma_2, lv_gamma_3;
 G4double weight = 0.0, random_weight = 0.0;
 G4DecayProducts* decay_products = nullptr;
 G4DynamicParticle* gamma_1 = nullptr;
 G4DynamicParticle* gamma_2 = nullptr;
 G4DynamicParticle* gamma_3 = nullptr;

 do
 {
  if ( decay_products != nullptr ) {  delete decay_products; }

  decay_products = G4GeneralPhaseSpaceDecay::DecayIt();

  gamma_1 = (*decay_products)[0];
  gamma_2 = (*decay_products)[1];
  gamma_3 = (*decay_products)[2];

  lv_gamma_1 = gamma_1->Get4Momentum();
  lv_gamma_2 = gamma_2->Get4Momentum();
  lv_gamma_3 = gamma_3->Get4Momentum();

  weight = GetOrthoPsM( lv_gamma_1.e(), lv_gamma_2.e(), lv_gamma_3.e() );
  random_weight = kOrthoPsMMax * G4UniformRand();
 }
 while( random_weight > weight );

 ///Polarization
 G4ThreeVector polarization_gamma_1 = GetPolarization( gamma_1->GetMomentumDirection() );
 G4ThreeVector polarization_gamma_2 = GetPolarization( gamma_2->GetMomentumDirection() );
 G4ThreeVector polarization_gamma_3 = GetPolarization( gamma_3->GetMomentumDirection() );

 gamma_1->SetPolarization( polarization_gamma_1.x(), polarization_gamma_1.y(), polarization_gamma_1.z() );
 gamma_2->SetPolarization( polarization_gamma_2.x(), polarization_gamma_2.y(), polarization_gamma_2.z() );
 gamma_3->SetPolarization( polarization_gamma_3.x(), polarization_gamma_3.y(), polarization_gamma_3.z() ); 

 return decay_products; 
}

G4double GatePositroniumDecayChannel::GetOrthoPsM( const G4double w1, const G4double w2, const G4double w3 ) const
{
 return pow( ( kElectronMass - w1 ) / ( w2 * w3 ), 2 ) + pow( ( kElectronMass - w2 ) / ( w1 * w3 ), 2 ) + pow( ( kElectronMass - w3 ) / ( w1 * w2 ), 2 );
}

G4ThreeVector GatePositroniumDecayChannel::GetPolarization( const G4ThreeVector& momentum ) const
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
  
G4ThreeVector GatePositroniumDecayChannel::GetPerpendicularVector(const G4ThreeVector& v) const
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
  





