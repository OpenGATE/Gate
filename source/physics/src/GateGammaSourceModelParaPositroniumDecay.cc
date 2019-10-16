/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateGammaSourceModelParaPositroniumDecay.hh"
#include "TLorentzVector.h"
#include "G4Electron.hh"
#include "G4PhysicalConstants.hh"
#include "GateConstants.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>

GateGammaSourceModelParaPositroniumDecay* GateGammaSourceModelParaPositroniumDecay::ptrJPETParaPositroniumDecayModel = 0;

GateGammaSourceModelParaPositroniumDecay::GateGammaSourceModelParaPositroniumDecay()
{
 SetParticlesNumber(2);
 SetGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel::ParaPositronium );
 GateExtendedVSourceManager::GetInstance()->AddGammaSourceModel(this);
}

GateGammaSourceModelParaPositroniumDecay::~GateGammaSourceModelParaPositroniumDecay() {}

void GateGammaSourceModelParaPositroniumDecay::GetGammaParticles(std::vector<G4PrimaryParticle*>& particles)
{

 Double_t mass_e = 0.511*MeV/1000.0;//GeV - because TGenPhaseSpace work with GeV

 TLorentzVector pozytonium(0.0, 0.0, 0.0, 2.0*mass_e);

 // 2 gamma quanta mass
 Double_t mass_secondaries[2] = {0.0, 0.0};

 TGenPhaseSpace m_2_body_decay;
 m_2_body_decay.SetDecay(pozytonium, 2, mass_secondaries);
 m_2_body_decay.Generate();

 int particles_number = GetParticlesNumber();

 if( particles_number != 2)
  G4cout<<"Incorrect number of particles. Number of particles: "<<particles_number<<G4endl;

 for(int i=0; i<particles_number; ++i){
  TLorentzVector partDir = *m_2_body_decay.GetDecay(i);
  partDir.Boost(GetPositronMomentum());
  particles[i]->SetMomentum( (partDir.Px())*1000.0, (partDir.Py())*1000.0, (partDir.Pz())*1000.0 ); // "*1000.0" because GetDecay return momentum in GeV but Geant4 and Gate make calculation in MeV
  particles[i]->SetPolarization(GetPolarization(particles[i]->GetMomentumDirection()));
 }

 particles[1]->SetPolarization( GetPerpendicularPolarizationToItsMomentumAndOtherPolarization( particles[1]->GetMomentum(), particles[0]->GetPolarization() ) );

 //Adding model info
 particles[0]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromParaPositronium, particles[0]->GetPolarization() ) );
 particles[1]->SetUserInformation( GetModelInfoForGamma( GateGammaModelPrimaryParticleInformation::GammaKind::GammaFromParaPositronium, particles[1]->GetPolarization() ) );

}

G4String GateGammaSourceModelParaPositroniumDecay::GetModelName() const
{
 return "pPs";
}

GateGammaSourceModelParaPositroniumDecay* GateGammaSourceModelParaPositroniumDecay::GetInstance()
{
 if(!ptrJPETParaPositroniumDecayModel){
  ptrJPETParaPositroniumDecayModel = new GateGammaSourceModelParaPositroniumDecay;
 }

 return ptrJPETParaPositroniumDecayModel;
}
