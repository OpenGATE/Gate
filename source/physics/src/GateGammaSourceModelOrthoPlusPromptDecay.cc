/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateGammaSourceModelOrthoPlusPromptDecay.hh"
#include "TGenPhaseSpace.h"
#include "TLorentzVector.h"
#include "G4Electron.hh"
#include "G4SystemOfUnits.hh"

GateGammaSourceModelOrthoPlusPromptDecay* GateGammaSourceModelOrthoPlusPromptDecay::ptrOrthoPlusPromptDecayModel = 0;

GateGammaSourceModelOrthoPlusPromptDecay::GateGammaSourceModelOrthoPlusPromptDecay()
{
 SetParticlesNumber( 4 );
 SetGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel::OrthoPositroniumAndPrompt );
 GateExtendedVSourceManager::GetInstance()->AddGammaSourceModel( this );
 //m_random_gen.SetSeed(0);
}

GateGammaSourceModelOrthoPlusPromptDecay::~GateGammaSourceModelOrthoPlusPromptDecay() {}

Double_t GateGammaSourceModelOrthoPlusPromptDecay::calculate_mQED( Double_t mass_e, Double_t w1, Double_t w2, Double_t w3 ) const
{
 return pow( ( mass_e - w1 ) / ( w2 * w3 ), 2 ) + pow( ( mass_e-w2 ) / ( w1 * w3 ), 2 ) + pow( ( mass_e - w3 ) / ( w1 * w2 ) , 2 );
}

void GateGammaSourceModelOrthoPlusPromptDecay::GetGammaParticles( std::vector<G4PrimaryParticle*>& particles )
{
 AddGammaFromDeexcitation( particles );
 AddGammasFromOrtoPositronium( particles );
}

void GateGammaSourceModelOrthoPlusPromptDecay::AddGammasFromOrtoPositronium( std::vector<G4PrimaryParticle*>& particles )
{
 Double_t mass_e = 0.511 * MeV / 1000.0;//GeV - because TGenPhaseSpace work with GeV

 TLorentzVector pozytonium( 0.0, 0.0, 0.0, 2.0 * mass_e );

 // 3 gamma quanta mass
 Double_t mass_secondaries[3] = { 0.0, 0.0, 0.0 };

 TGenPhaseSpace m_3_body_decay;
 m_3_body_decay.SetDecay( pozytonium, 3, mass_secondaries );

 // Include dacay's weights
 Double_t weight;
 Double_t weight_max= m_3_body_decay.GetWtMax() * pow( 10, 5 );
 Double_t rwt;
 Double_t M_max = 7.65928 * pow( 10,-6 );

 do {
  weight = m_3_body_decay.Generate();
  weight = weight*calculate_mQED( mass_e, m_3_body_decay.GetDecay( 0 )->E(), m_3_body_decay.GetDecay(1)->E(), m_3_body_decay.GetDecay(2)->E() );
  rwt = m_random_gen.Uniform( M_max * weight_max );
 } while( rwt > weight );

 int particles_number = GetParticlesNumber();

 for(int i = 1; i < particles_number; ++i){
  TLorentzVector partDir = *m_3_body_decay.GetDecay( i - 1 );
  partDir.Boost( GetPositronMomentum() );
  // "*1000.0" because GetDecay return momentum in GeV but Geant4 and Gate make calculation in MeV
  particles[ i ]->SetMomentum( ( partDir.Px() ) * 1000.0, (partDir.Py() ) * 1000.0, ( partDir.Pz() ) * 1000.0 ); 
  particles[ i ]->SetPolarization( GetPolarization( particles[ i ]->GetMomentumDirection() ) );
 }

 //Adding model info
 particles[ 1 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromOrthoPositronium, particles[ 1 ]->GetPolarization() ) );
 particles[ 2 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromOrthoPositronium, particles[ 2 ]->GetPolarization() ) );
 particles[ 3 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromOrthoPositronium, particles[ 3 ]->GetPolarization() ) );
}

void GateGammaSourceModelOrthoPlusPromptDecay::AddGammaFromDeexcitation( std::vector<G4PrimaryParticle*>& particles )
{
 G4ThreeVector momentum_direction = GetRandomVectorOnSphere();
 G4double kinetic_energy = GetPromptGammaEnergy();
 G4ThreeVector gamma_polarization = GetPolarization( momentum_direction );

 G4ThreeVector momentum = kinetic_energy * momentum_direction;

 particles[ 0 ]->Set4Momentum( momentum.x(), momentum.y(), momentum.z(), kinetic_energy );
 particles[ 0 ]->SetPolarization( gamma_polarization );

 //Adding model info
 particles[ 0 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaPrompt, particles[ 0 ]->GetPolarization() ) );
}


G4ThreeVector GateGammaSourceModelOrthoPlusPromptDecay::GetRandomVectorOnSphere()
{
 Double_t x = 0, y = 0, z = 0;
 GetRandomGenerator()->Sphere( x, y, z, 1.0 );
 return G4ThreeVector( x, y, z );
}

G4String GateGammaSourceModelOrthoPlusPromptDecay::GetModelName() const
{
 return "oPsAndPrompt";
}

GateGammaSourceModelOrthoPlusPromptDecay* GateGammaSourceModelOrthoPlusPromptDecay::GetInstance()
{
 if ( !ptrOrthoPlusPromptDecayModel )
  ptrOrthoPlusPromptDecayModel = new GateGammaSourceModelOrthoPlusPromptDecay();
 return ptrOrthoPlusPromptDecayModel;
}
