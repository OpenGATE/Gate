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

#include "GatePositroniumDecayModel.hh"

int GatePositroniumDecayModel::getPositroniumDecayIndex(const std::vector<float>& fractions) {
  auto r = G4UniformRand(); 
  float curr_frac_cumulative = 0.0;
  for (int i = 0; i < fractions.size(); ++i) {
    curr_frac_cumulative = curr_frac_cumulative + fractions[i];
    if(r<= curr_frac_cumulative) return i;   
 }
  return -1;
}

GatePositroniumDecayModel::GatePositroniumDecayModel(const PositroniumDecayModelParams& modelParams):fModelParams(modelParams)
{
  auto num_of_decay_channels = fModelParams.fDecayKind.size();
  for (int i = 0; i < num_of_decay_channels; i++) {
    if (fModelParams.fDecayKind[i] == PositroniumDecayKind::k2Gamma) 
    {
      fPositroniumDecayChannel.push_back(std::move(GatePositronium("pPs", fModelParams.fLifetimes[i]* ns)));
    } else {
      fPositroniumDecayChannel.push_back(std::move(GatePositronium("oPs", fModelParams.fLifetimes[i]* ns)));
    }
  }
}

G4PrimaryVertex* GatePositroniumDecayModel::GetPrimaryVertexFromPositroniumAnnihilation(G4double particle_time, const G4ThreeVector& particle_position, int decayIndex)
{

 G4double shifted_particle_time = particle_time + G4RandExponential::shoot(fModelParams.fLifetimes[decayIndex]);

 G4PrimaryVertex* vertex = new G4PrimaryVertex( particle_position, shifted_particle_time );
 std::vector<G4PrimaryParticle*> gammas = GetGammasFromPositroniumAnnihilation(decayIndex);
 std::for_each( gammas.begin(), gammas.end(), [&]( G4PrimaryParticle* gamma ) { vertex->SetPrimary( gamma ); } );
 return vertex;
}

G4int GatePositroniumDecayModel::GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position)
{
  auto decayIndex = GatePositroniumDecayModel::getPositroniumDecayIndex(fModelParams.fFractions);

  G4int number_of_vertices = 1;
  if(fModelParams.fPromptGammaProbabilities[decayIndex] > G4UniformRand()) 
  { 
    ++number_of_vertices;
    event->AddPrimaryVertex(GetPrimaryVertexFromDeexcitation(particle_time, particle_position, decayIndex)); 
  }
  event->AddPrimaryVertex(GetPrimaryVertexFromPositroniumAnnihilation(particle_time, particle_position, decayIndex));
  return number_of_vertices;
} 

G4PrimaryParticle* GatePositroniumDecayModel::GetGammaFromDeexcitation(int decayIndex)
{
 G4PrimaryParticle* gamma = GetSingleGamma(fModelParams.fPromptGammaEnergy[decayIndex]);
 gamma->SetUserInformation( GetPrimaryParticleInformation( gamma, GateEmittedGammaInformation::GammaKind::Prompt ) );
 return gamma;
}

G4PrimaryVertex* GatePositroniumDecayModel::GetPrimaryVertexFromDeexcitation(G4double particle_time, const  G4ThreeVector& particle_position, int decayIndex)
{
 G4PrimaryVertex* vertex = new G4PrimaryVertex(particle_position, particle_time);
 vertex->SetPrimary(GetGammaFromDeexcitation(decayIndex));
 return vertex;
}

std::vector<G4PrimaryParticle*> GatePositroniumDecayModel::GetGammasFromPositroniumAnnihilation(int decayIndex)
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

