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

void GatePositroniumSource::PrepareModel() 
{
  if (GetType() != "Ps") {
    GateError("GatePositroniumSource::PrepareModel - model type is not Positronium. Current type: " + GetType());
  }

  auto params = pMessenger->generatePositroniumDecayParams();
  pModel = std::make_unique<GatePositroniumDecayModel>(params);
}

G4int GatePositroniumSource::GeneratePrimaries(G4Event* event)
{
  
 if (!pModel) { PrepareModel(); }

 G4double particle_time = GetTime();
 G4ThreeVector particle_position = GetPosDist()->GenerateOne();
 ChangeParticlePositionRelativeToAttachedVolume(particle_position);
 return pModel->GeneratePrimaryVertices(event, particle_time, particle_position);
}

