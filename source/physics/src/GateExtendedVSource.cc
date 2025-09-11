/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateExtendedVSource.hh"
#include "GatePositroniumDecayModel.hh"
#include "G4Event.hh"

#include <map>

GateExtendedVSource::GateExtendedVSource(const G4String& name ) : GateVSource( name )
{
 pMessenger =  new GateExtendedVSourceMessenger( this );
}

GateExtendedVSource::~GateExtendedVSource()
{
 if ( pMessenger != nullptr ) { delete pMessenger; } 
 if ( pModel != nullptr ) { delete pModel; }
}

void GateExtendedVSource::SetModel(const G4String &model_name) 
{
  static const std::map<G4String, ModelKind> models{
      {"sg", GateExtendedVSource::ModelKind::SingleGamma},
      {"pPs", GateExtendedVSource::ModelKind::ParaPositronium},
      {"oPs", GateExtendedVSource::ModelKind::OrthoPositronium},
      {"Ps", GateExtendedVSource::ModelKind::Positronium}};

  auto it = models.find(model_name);
  if (it != models.end())
  {
    fModelKind = it->second;
  } else {
    fBehaveLikeVSource = true;
    G4cout << "GateExtendedVSource::SetModel : Unknown gamma source model. "
              "Enable: sg, pPs, oPs, Ps. Switching to GateVSource behavour."
           << G4endl;
  }
}

void GateExtendedVSource::SetEnableDeexcitation(G4bool enable_deexcitation) { fEnableDeexcitation =  enable_deexcitation; }

void GateExtendedVSource::SetFixedEmissionDirection(const G4ThreeVector& fixed_emission_direction) { fFixedEmissionDirection = fixed_emission_direction; }

void GateExtendedVSource::SetEnableFixedEmissionDirection(G4bool enable_fixed_emission_direction) { fEnableFixedEmissionDirection = enable_fixed_emission_direction; }

void GateExtendedVSource::SetEmissionEnergy(G4double energy) { fEmissionEnergy =energy; }

void GateExtendedVSource::SetSeed(G4long seed) { fSeed = seed; }

void GateExtendedVSource::SetPostroniumLifetime(const G4String& positronium_name, G4double life_time) 
{
 if ( positronium_name == kParaPositroniumName ) { fParaPostroniumLifetime = life_time; }
 else if ( positronium_name == kOrthoPositroniumName ) { fOrthoPostroniumLifetime =  life_time; }
 else { GateError( "GateExtendedVSource::SetPostroniumLifetime : incorrect positronium name - try: pPs or oPs" ); } 
}

void GateExtendedVSource::SetPromptGammaEnergy(G4double energy) { fPromptGammaEnergy = energy; }

void GateExtendedVSource::SetPositroniumFraction( const G4String& positronium_kind, G4double fraction )
{
 if ( fraction > 1.0 || fraction < 0.0 )
 {
  GateError( "GateExtendedVSource::SetPositroniumFraction : incorrect fraction value - required: 0.0 <= fraction <= 1.0 " ); 
 }

 G4double pPs_fraction = 0.0;
 
 if ( positronium_kind == kParaPositroniumName ) { pPs_fraction = fraction; }
 else if ( positronium_kind == kOrthoPositroniumName ) { pPs_fraction = 1.0 - fraction; }
 else { GateError( "GateExtendedVSource::SetPositroniumFraction : incorrect positronium kind - enable are: pPs, oPs" ); }

 fParaPositroniumFraction =  pPs_fraction;
}

void GateExtendedVSource::PrepareModel()
{
 SetModel( GetType() );

 if ( fBehaveLikeVSource )  return;

   
 if ( fModelKind == GateExtendedVSource::ModelKind::ParaPositronium || fModelKind == GateExtendedVSource::ModelKind::OrthoPositronium || fModelKind == GateExtendedVSource::ModelKind::Positronium )
 {
  pModel = new GatePositroniumDecayModel();
  GatePositroniumDecayModel* model = dynamic_cast<GatePositroniumDecayModel*>( pModel );

  if ( fModelKind == GateExtendedVSource::ModelKind::OrthoPositronium ) { model->SetPositroniumKind( GatePositroniumDecayModel::PositroniumKind::oPs ); }
  if ( fModelKind == GateExtendedVSource::ModelKind::Positronium && fParaPositroniumFraction.has_value() ) { model->SetParaPositroniumFraction( fParaPositroniumFraction.value() ); }

  if ( fEnableDeexcitation.has_value() && fEnableDeexcitation.value() ) { model->SetDecayModel( GatePositroniumDecayModel::DecayModel::WithPrompt ); } 
  if ( fParaPostroniumLifetime.has_value() ) { model->SetPostroniumLifetime( kParaPositroniumName, fParaPostroniumLifetime.value() ); }
  if ( fOrthoPostroniumLifetime.has_value() ) { model->SetPostroniumLifetime( kOrthoPositroniumName, fOrthoPostroniumLifetime.value() ); }
  if ( fPromptGammaEnergy.has_value() ) { model->SetPromptGammaEnergy( fPromptGammaEnergy.value() ); }
 }
 else if ( fModelKind == GateExtendedVSource::ModelKind::SingleGamma ) { pModel = new GateGammaEmissionModel(); }
 else { GateError( "GateExtendedVSource::PrepareModel - unknown model." ); }

 if ( fFixedEmissionDirection.has_value() ) { pModel->SetFixedEmissionDirection( fFixedEmissionDirection.value() ); }
 if ( fEnableFixedEmissionDirection.has_value() ) { pModel->SetEnableFixedEmissionDirection( fEnableFixedEmissionDirection.value() ); }
 if ( fEmissionEnergy.has_value() ) { pModel->SetEmissionEnergy( fEmissionEnergy.value() ); }
 if ( fSeed.has_value() ) { pModel->SetSeed( fSeed.value() ); }

}

G4int GateExtendedVSource::GeneratePrimaries(G4Event* event)
{
 if (!fBehaveLikeVSource && !pModel) { PrepareModel(); }
 if (fBehaveLikeVSource) { return GateVSource::GeneratePrimaries(event); }
 
 G4double particle_time = GetTime();
 G4ThreeVector particle_position = GetPosDist()->GenerateOne();
 ChangeParticlePositionRelativeToAttachedVolume(particle_position);
 return pModel->GeneratePrimaryVertices(event, particle_time, particle_position);
}

