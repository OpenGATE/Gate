/**
 *  @copyright Copyright 2018 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @file GateGammaSourceModelParaPlusPromptDecay.cc
 */

#include "GateGammaSourceModelParaPlusPromptDecay.hh"
#include "TLorentzVector.h"
#include "G4Electron.hh"
#include "G4PhysicalConstants.hh"
#include "GateConstants.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>

GateGammaSourceModelParaPlusPromptDecay* GateGammaSourceModelParaPlusPromptDecay::ptrJPETParaPlusPromptDecayModel=nullptr;

GateGammaSourceModelParaPlusPromptDecay::GateGammaSourceModelParaPlusPromptDecay()
{
 SetParticlesNumber(3);
 SetGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel::ParaPositroniumAndPrompt );
 GateExtendedVSourceManager::GetInstance()->AddGammaSourceModel(this);
}

GateGammaSourceModelParaPlusPromptDecay::~GateGammaSourceModelParaPlusPromptDecay()
{
}

void GateGammaSourceModelParaPlusPromptDecay::GetGammaParticles(std::vector<G4PrimaryParticle*>& particles)
{
 AddGammaFromDeexcitation( particles );
 AddGammasFromParaPositronium( particles );
}

G4String GateGammaSourceModelParaPlusPromptDecay::GetModelName() const
{
 return "pPsAndPrompt";
}

GateGammaSourceModelParaPlusPromptDecay* GateGammaSourceModelParaPlusPromptDecay::GetInstance()
{
 if(!ptrJPETParaPlusPromptDecayModel)
  ptrJPETParaPlusPromptDecayModel = new GateGammaSourceModelParaPlusPromptDecay;
 return ptrJPETParaPlusPromptDecayModel;
}


void GateGammaSourceModelParaPlusPromptDecay::AddGammaFromDeexcitation( std::vector<G4PrimaryParticle*>& particles )
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

void GateGammaSourceModelParaPlusPromptDecay::AddGammasFromParaPositronium( std::vector<G4PrimaryParticle*>& particles )
{
 Double_t mass_e = 0.511 * MeV / 1000.0;//GeV - because TGenPhaseSpace work with GeV
 TLorentzVector positronium( 0.0, 0.0, 0.0, 2.0 * mass_e );
 Double_t mass_secondaries[ 2 ] = { 0.0, 0.0 };

 TGenPhaseSpace two_body_decay;
 two_body_decay.SetDecay( positronium, 2, mass_secondaries );
 two_body_decay.Generate();

 TLorentzVector gamma_1_momentum = *two_body_decay.GetDecay( 0 );
 particles[ 1 ]->SetMomentum( gamma_1_momentum.Px() * 1000.0, gamma_1_momentum.Py() * 1000.0, gamma_1_momentum.Pz() * 1000.0 );

 G4ThreeVector gamma_1_polarization = GetPolarization( particles[ 1 ]->GetMomentumDirection() );
 particles[ 1 ]->SetPolarization( gamma_1_polarization );

 TLorentzVector gamma_2_momentum = *two_body_decay.GetDecay( 1 );
 particles[ 2 ]->SetMomentum( gamma_2_momentum.Px() * 1000.0, gamma_2_momentum.Py() * 1000.0, gamma_2_momentum.Pz() * 1000.0 );

 G4ThreeVector gamma_2_polarization = GetPerpendicularPolarizationToItsMomentumAndOtherPolarization( particles[ 2 ]->GetMomentum(), gamma_1_polarization );
 particles[ 2 ]->SetPolarization( gamma_2_polarization );

 //Adding model info
 particles[ 1 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromParaPositronium, particles[ 1 ]->GetPolarization() ) );
 particles[ 2 ]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromParaPositronium, particles[ 2 ]->GetPolarization() ) );
}

G4ThreeVector GateGammaSourceModelParaPlusPromptDecay::GetRandomVectorOnSphere()
{
 Double_t x = 0, y = 0, z = 0;
 GetRandomGenerator()->Sphere( x, y, z, 1.0 );
 return G4ThreeVector( x, y, z );
}
