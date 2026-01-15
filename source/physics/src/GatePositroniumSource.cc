/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <map>

#include "G4Event.hh"

#include "GatePositroniumSource.hh"
#include "GatePositroniumDecayModel.hh"


GatePositroniumSource::GatePositroniumSource(const G4String &name)
    : GateVSource(name),
      pMessenger(std::make_unique<GatePositroniumSourceMessenger>(this)) 
{
}

void GatePositroniumSource::SetModel(const G4String &model_name) 
{
  static const std::map<G4String, ModelKind> models{
      {"pPs", GatePositroniumSource::ModelKind::ParaPositronium},
      {"oPs", GatePositroniumSource::ModelKind::OrthoPositronium},
      {"Ps", GatePositroniumSource::ModelKind::Positronium}};

  auto it = models.find(model_name);
  if (it != models.end())
  {
    fModelKind = it->second;
  } else {
    fBehaveLikeVSource = true;
    G4cout << "GatePositroniumSource::SetModel : Unknown gamma source model. "
              "Enable: pPs, oPs, Ps. Switching to GateVSource behavour."
           << G4endl;
  }
}

void GatePositroniumSource::PrepareModel() 
{
  SetModel(GetType());

  if (fBehaveLikeVSource) {
    return;
  }

  if (fModelKind == GatePositroniumSource::ModelKind::Positronium) {
    auto params = pMessenger->generatePositroniumDecayParams();
    pModel = std::make_unique<PositroniumDecayModel>(params);
  } else {
    if (fModelKind == GatePositroniumSource::ModelKind::ParaPositronium) {
      auto params = pMessenger->generatePositroniumDecayParams(
          GatePositroniumDecayParamsGenerator::kParaPositronium);
      pModel = std::make_unique<PositroniumDecayModel>(params);

    } else {
      if (fModelKind == GatePositroniumSource::ModelKind::OrthoPositronium) {
        auto params = pMessenger->generatePositroniumDecayParams(
            GatePositroniumDecayParamsGenerator::kOrthoPositronium);
        pModel = std::make_unique<PositroniumDecayModel>(params);
      } else {
        GateError("GatePositroniumSource::PrepareModel - unknown model.");
      }
    }
  }
}

G4int GatePositroniumSource::GeneratePrimaries(G4Event* event)
{
 if (!fBehaveLikeVSource && !pModel) { PrepareModel(); }
 if (fBehaveLikeVSource) { return GateVSource::GeneratePrimaries(event); }
 
 G4double particle_time = GetTime();
 G4ThreeVector particle_position = GetPosDist()->GenerateOne();
 ChangeParticlePositionRelativeToAttachedVolume(particle_position);
 return pModel->GeneratePrimaryVertices(event, particle_time, particle_position);
}

