/**
 *  @copyright Copyright 2017 The J-PET Gate Authors. All rights reserved.
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
 *  @file GateGammaSourceModelOrthoPositroniumDecay.cc
 */

#include "GateGammaSourceModelOrthoPositroniumDecay.hh"
#include "TGenPhaseSpace.h"
#include "TLorentzVector.h"
#include "G4Electron.hh"
#include "G4SystemOfUnits.hh"

GateGammaSourceModelOrthoPositroniumDecay* GateGammaSourceModelOrthoPositroniumDecay::ptrJPETOrtoPositroniumDecayModel = 0;

GateGammaSourceModelOrthoPositroniumDecay::GateGammaSourceModelOrthoPositroniumDecay()
{
 SetParticlesNumber(3);
 SetGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel::OrthoPositronium );
 GateExtendedVSourceManager::GetInstance()->AddGammaSourceModel(this);
}

GateGammaSourceModelOrthoPositroniumDecay::~GateGammaSourceModelOrthoPositroniumDecay() {}

Double_t GateGammaSourceModelOrthoPositroniumDecay::calculate_mQED(Double_t mass_e, Double_t w1, Double_t w2, Double_t w3) const
{
 return pow((mass_e-w1)/(w2*w3),2) + pow((mass_e-w2)/(w1*w3),2) + pow((mass_e-w3)/(w1*w2),2);
}

void GateGammaSourceModelOrthoPositroniumDecay::GetGammaParticles( std::vector<G4PrimaryParticle*>& particles )
{
 Double_t mass_e = 0.511*MeV/1000.0;//GeV - because TGenPhaseSpace work with GeV

 TLorentzVector pozytonium(0.0, 0.0, 0.0, 2.0*mass_e);

 // 3 gamma quanta mass
 Double_t mass_secondaries[3] = {0.0, 0.0, 0.0};

 TGenPhaseSpace m_3_body_decay;
 m_3_body_decay.SetDecay( pozytonium, 3, mass_secondaries );

 // Include dacay's weights
 Double_t weight;
 Double_t weight_max= m_3_body_decay.GetWtMax()*pow(10,5);
 Double_t rwt;
 Double_t M_max = 7.65928*pow(10,-6);

 do 
 {
  weight = m_3_body_decay.Generate();
  weight = weight*calculate_mQED(mass_e,m_3_body_decay.GetDecay(0)->E(),m_3_body_decay.GetDecay(1)->E(),m_3_body_decay.GetDecay(2)->E());
  rwt = m_random_gen.Uniform(M_max*weight_max);
 }
 while( rwt > weight );


 int particles_number = GetParticlesNumber();
 for ( int i = 0; i < particles_number; ++i ) 
 {
  TLorentzVector partDir = *m_3_body_decay.GetDecay(i);
  partDir.Boost(GetPositronMomentum());
  particles[i]->SetMomentum( (partDir.Px())*1000.0, (partDir.Py())*1000.0, (partDir.Pz())*1000.0 ); // "*1000.0" because GetDecay return momentum in GeV but Geant4 and Gate make calculation in MeV
  particles[i]->SetPolarization(GetPolarization(particles[i]->GetMomentumDirection()));
 }

 //Adding model info
 particles[0]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromOrthoPositronium, particles[0]->GetPolarization() ) );
 particles[1]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromOrthoPositronium, particles[1]->GetPolarization() ) );
 particles[2]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromOrthoPositronium, particles[2]->GetPolarization() ) );
}

G4String GateGammaSourceModelOrthoPositroniumDecay::GetModelName() const { return "oPs"; }

GateGammaSourceModelOrthoPositroniumDecay* GateGammaSourceModelOrthoPositroniumDecay::GetInstance()
{
 if ( !ptrJPETOrtoPositroniumDecayModel ) { ptrJPETOrtoPositroniumDecayModel = new GateGammaSourceModelOrthoPositroniumDecay(); }
 
 return ptrJPETOrtoPositroniumDecayModel;
}
