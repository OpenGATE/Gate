/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <cmath>
#include <algorithm>
#include <cassert>

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
  return static_cast<int>(fractions.size()) - 1;
}

G4ThreeVector GatePositroniumDecayModel::AddPositronRangeShift(const G4ThreeVector& original_position,  G4double mean_positron_range)
{
  // r = sqrt(x**2+y**2+z**2)
  // <r> = sigma * sqrt(8/Pi) // matching mean for 3-D Gaussian
  const G4double sqrt8_over_pi = std::sqrt(8.0/CLHEP::pi);
  G4double sigma  = mean_positron_range/sqrt8_over_pi ;
  G4ThreeVector shift(G4RandGauss::shoot(0., sigma),
                      G4RandGauss::shoot(0., sigma),
                      G4RandGauss::shoot(0., sigma));
  return original_position + shift;
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
 bool is_positron_range_enabled = fModelParams.fMeanPositronRangeEnabled[decayIndex]; 

 G4double shifted_particle_time = particle_time + G4RandExponential::shoot(fModelParams.fLifetimes[decayIndex]);

 auto shifted_particle_position = particle_position;
 if (is_positron_range_enabled)
 {
   shifted_particle_position = AddPositronRangeShift(particle_position, fModelParams.fMeanPositronRange[decayIndex]); 
}

 G4PrimaryVertex* vertex = new G4PrimaryVertex( shifted_particle_position, shifted_particle_time );
 std::vector<G4PrimaryParticle*> gammas = GetGammasFromPositroniumAnnihilation(decayIndex);
 std::for_each( gammas.begin(), gammas.end(), [&]( G4PrimaryParticle* gamma ) { vertex->SetPrimary( gamma ); } );
 return vertex;
}

G4int GatePositroniumDecayModel::GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position)
{
  G4int number_of_vertices = 0;
  while (number_of_vertices <=0) { 
    auto decayIndex = GatePositroniumDecayModel::getPositroniumDecayIndex(fModelParams.fFractions);
    auto no_electron_capture_prob = 1- fModelParams.fElectronCaptureProbabilities[decayIndex];
    assert(no_electron_capture_prob>=0);

    if(G4UniformRand() <= fModelParams.fPromptGammaProbabilities[decayIndex]) 
    { 
      ++number_of_vertices;
      event->AddPrimaryVertex(GetPrimaryVertexFromDeexcitation(particle_time, particle_position, decayIndex)); 
    }

    if(G4UniformRand() <= no_electron_capture_prob) 
    {
      ++number_of_vertices;
      event->AddPrimaryVertex(GetPrimaryVertexFromPositroniumAnnihilation(particle_time, particle_position, decayIndex));
    }
  }
  return number_of_vertices;
} 

GateEmittedGammaInformation::DecayModel GatePositroniumDecayModel::GetDecayModel(const int decayIndex) const
{
  if (fModelParams.fPromptGammaProbabilities[decayIndex] > 0) {
    return GateEmittedGammaInformation::DecayModel::Deexcitation;
  }
  return GateEmittedGammaInformation::DecayModel::Standard;
}

GateEmittedGammaInformation::SourceKind GatePositroniumDecayModel::GetSourceKind(int decayIndex) const
{
  // fPositronInteractions is only populated when setPositronInteractions is explicitly called.
  // We intentionally do not fall back to inferring SourceKind from fDecayKind because
  // k2Gamma does not uniquely identify pPs: oPs pick-off/quenching also produces 2 gammas.
  if (decayIndex >= static_cast<int>(fModelParams.fPositronInteractions.size()))
    return GateEmittedGammaInformation::SourceKind::NotDefined;

  const PositronElectronInteraction interaction = fModelParams.fPositronInteractions[decayIndex];
  switch (interaction)
  {
    case PositronElectronInteraction::kParaPs:
      return GateEmittedGammaInformation::SourceKind::ParaPositronium;
    case PositronElectronInteraction::kOrthoPs:
      return GateEmittedGammaInformation::SourceKind::OrthoPositronium;
    case PositronElectronInteraction::kDirect:
      return GateEmittedGammaInformation::SourceKind::DirectAnnihilation;
    default:
      break;
  }
  return GateEmittedGammaInformation::SourceKind::NotDefined;
}

G4PrimaryParticle* GatePositroniumDecayModel::GetGammaFromDeexcitation(int decayIndex)
{
 G4PrimaryParticle* gamma = GetSingleGamma(fModelParams.fPromptGammaEnergy[decayIndex]);
 GateEmittedGammaInformation* info = GetPrimaryParticleInformation( gamma, GateEmittedGammaInformation::GammaKind::Prompt );
 info->SetDecayIndex( decayIndex );
 info->SetDecayModel( GetDecayModel(decayIndex) );
 auto sourceKind = GetSourceKind(decayIndex);
 if (sourceKind != GateEmittedGammaInformation::SourceKind::NotDefined)
  info->SetSourceKind( sourceKind );
 gamma->SetUserInformation( info );
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
  GateEmittedGammaInformation* info = GetPrimaryParticleInformation( gamma, GateEmittedGammaInformation::GammaKind::Annihilation );
  info->SetDecayIndex( decayIndex );
  info->SetDecayModel( GetDecayModel(decayIndex) );
  auto sourceKind = GetSourceKind(decayIndex);
  if (sourceKind != GateEmittedGammaInformation::SourceKind::NotDefined)
   info->SetSourceKind( sourceKind );
  gamma->SetUserInformation( info );
  gammas[i] = gamma;
 }
 delete decay_products;

 return gammas;
}

