/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "G4DecayProducts.hh"
#include "G4ParticleTable.hh"
#include "G4DecayTable.hh"
#include "G4ParticleDefinition.hh"

#include "GatePositronium.hh"

GatePositronium::GatePositronium(const G4String& name, G4double life_time): fName(name), fLifeTime(life_time)
{
  G4ParticleDefinition *positronium_def = G4ParticleTable::GetParticleTable()->FindParticle(name);
  G4DecayTable *positronium_decay_table = positronium_def->GetDecayTable();
  pDecayChannel = static_cast<GatePositroniumDecayChannel*>(positronium_decay_table->GetDecayChannel(0));
}

G4double GatePositronium::GetLifeTime() const { return fLifeTime; }

const G4String& GatePositronium::GetName() const { return fName; }

G4int GatePositronium::GetAnnihilationGammasNumber() const { return pDecayChannel->GetNumberOfDaughters(); }

G4DecayProducts* GatePositronium::GetDecayProducts() const { return pDecayChannel->DecayIt(); }
