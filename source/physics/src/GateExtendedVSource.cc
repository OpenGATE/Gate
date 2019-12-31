/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateExtendedVSource.hh"
#include <algorithm>
#include "GatePositroniumDecayModel.hh"
#include "G4Event.hh"

GateExtendedVSource::GateExtendedVSource( G4String name ) : GateVSource( name )
{
 pMessenger =  new GateExtendedVSourceMessenger( this );
}

GateExtendedVSource::~GateExtendedVSource()
{
 if ( pMessenger != nullptr ) { delete pMessenger; } 
 if ( pModel != nullptr ) { delete pModel; }
}

void GateExtendedVSource::SetModel( const G4String& model_name )
{
 if ( model_name == "sg" ) { fModelKind = GateExtendedVSource::ModelKind::SingleGamma; }
 else if ( model_name == "pPs" ) { fModelKind = GateExtendedVSource::ModelKind::ParaPositronium; }
 else if ( model_name == "oPs" ) { fModelKind = GateExtendedVSource::ModelKind::OrthoPositronium; }
 else if ( model_name == "Ps" ) { fModelKind = GateExtendedVSource::ModelKind::Positronium; }
 else 
 { 
  fBehaveLikeVSource = true;
  G4cout << "GateExtendedVSource::SetModel : Unknown gamma source model. Enable: sg, pPs, oPs, Ps. Switching to GateVSource behavour." << G4endl; }
}

void GateExtendedVSource::SetEnableDeexcitation( const G4bool& enable_deexcitation ) { fEnableDeexcitation.Set( enable_deexcitation ); }

void GateExtendedVSource::SetFixedEmissionDirection( const G4ThreeVector& fixed_emission_direction ) { fFixedEmissionDirection.Set( fixed_emission_direction ); }

void GateExtendedVSource::SetEnableFixedEmissionDirection( const G4bool& enable_fixed_emission_direction ) { fEnableFixedEmissionDirection.Set( enable_fixed_emission_direction ); }

void GateExtendedVSource::SetEmissionEnergy( const G4double& energy ) { fEmissionEnergy.Set( energy ); }

void GateExtendedVSource::SetSeed( const G4long& seed ) { fSeed.Set( seed ); }

void GateExtendedVSource::SetPostroniumLifetime( const G4String& positronium_name, const G4double& life_time ) 
{
 if ( positronium_name == kParaPositroniumName ) { fParaPostroniumLifetime.Set( life_time ); }
 else if ( positronium_name == kOrthoPositroniumName ) { fOrthoPostroniumLifetime.Set( life_time ); }
 else { GateError( "GateExtendedVSource::SetPostroniumLifetime : incorrect positronium name - try: pPs or oPs" ); } 
}

void GateExtendedVSource::SetPromptGammaEnergy( const G4double& energy ) { fPromptGammaEnergy.Set( energy ); }

void GateExtendedVSource::SetPositroniumFraction( const G4String& positronium_kind, const G4double& fraction )
{
 if ( fraction > 1.0 || fraction < 0.0 )
 {
  GateError( "GateExtendedVSource::SetPositroniumFraction : incorrect fraction value - required: 0.0 <= fraction <= 1.0 " ); 
 }

 G4double pPs_fraction = 0.0;
 
 if ( positronium_kind == kParaPositroniumName ) { pPs_fraction = fraction; }
 else if ( positronium_kind == kOrthoPositroniumName ) { pPs_fraction = 1.0 - fraction; }
 else { GateError( "GateExtendedVSource::SetPositroniumFraction : incorrect positronium kind - enable are: pPs, oPs" ); }

 fParaPositroniumFraction.Set( pPs_fraction );
}

void GateExtendedVSource::PrepareModel()
{
 SetModel( GetType() );

 if ( fBehaveLikeVSource ) { return; }

 if ( fModelKind == GateExtendedVSource::ModelKind::ParaPositronium || fModelKind == GateExtendedVSource::ModelKind::OrthoPositronium || fModelKind == GateExtendedVSource::ModelKind::Positronium )
 {
  pModel = new GatePositroniumDecayModel();
  GatePositroniumDecayModel* model = dynamic_cast<GatePositroniumDecayModel*>( pModel );

  if ( fModelKind == GateExtendedVSource::ModelKind::OrthoPositronium ) { model->SetPositroniumKind( GatePositroniumDecayModel::PositroniumKind::oPs ); }
  if ( fModelKind == GateExtendedVSource::ModelKind::Positronium && fParaPositroniumFraction.IsSetted() ) { model->SetParaPositroniumFraction( fParaPositroniumFraction.Get() ); }

  if ( fEnableDeexcitation.IsSetted() && fEnableDeexcitation.Get() ) { model->SetDecayModel( GatePositroniumDecayModel::DecayModel::WithPrompt ); } 
  if ( fParaPostroniumLifetime.IsSetted() ) { model->SetPostroniumLifetime( kParaPositroniumName, fParaPostroniumLifetime.Get() ); }
  if ( fOrthoPostroniumLifetime.IsSetted() ) { model->SetPostroniumLifetime( kOrthoPositroniumName, fOrthoPostroniumLifetime.Get() ); }
  if ( fPromptGammaEnergy.IsSetted() ) { model->SetPromptGammaEnergy( fPromptGammaEnergy.Get() ); }
 }
 else if ( fModelKind == GateExtendedVSource::ModelKind::SingleGamma ) { pModel = new GateGammaEmissionModel(); }
 else { GateError( "GateExtendedVSource::PrepareModel - unknown model." ); }

 if ( fFixedEmissionDirection.IsSetted() ) { pModel->SetFixedEmissionDirection( fFixedEmissionDirection.Get() ); }
 if ( fEnableFixedEmissionDirection.IsSetted() ) { pModel->SetEnableFixedEmissionDirection( fEnableFixedEmissionDirection.Get() ); }
 if ( fEmissionEnergy.IsSetted() ) { pModel->SetEmissionEnergy( fEmissionEnergy.Get() ); }
 if ( fSeed.IsSetted() ) { pModel->SetSeed( fSeed.Get() ); }

}

G4int GateExtendedVSource::GeneratePrimaries( G4Event* event )
{
 if ( !fBehaveLikeVSource && pModel == nullptr ) { PrepareModel(); }
 if ( fBehaveLikeVSource ) { return GateVSource::GeneratePrimaries( event ); }
 
 G4double particle_time = GetTime();
 G4ThreeVector particle_position = GetPosDist()->GenerateOne();
 ChangeParticlePositionRelativeToAttachedVolume( particle_position );
 return pModel->GeneratePrimaryVertices( event, particle_time, particle_position);
}

