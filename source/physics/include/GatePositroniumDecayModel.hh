/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositroniumDecayModel_hh
#define GatePositroniumDecayModel_hh

#include<vector>

#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"

#include "GateGammaEmissionModel.hh"
#include "GatePositroniumDecayModelParams.hh"
#include "GatePositronium.hh"

class GatePositroniumDecayModel:public GateGammaEmissionModel
{
  public:
  static int getPositroniumDecayIndex(const std::vector<float>& fractions);

  explicit GatePositroniumDecayModel(const PositroniumDecayModelParams& modelParams);

  protected:
  virtual G4int GeneratePrimaryVertices(G4Event* event, G4double& particle_time,  G4ThreeVector& particle_position) override;
  G4PrimaryVertex* GetPrimaryVertexFromDeexcitation(G4double particle_time, const  G4ThreeVector& particle_position, int decayIndex);
  G4PrimaryVertex *GetPrimaryVertexFromPositroniumAnnihilation(G4double particle_time, const G4ThreeVector &particle_position, int decayIndex);
  G4PrimaryParticle* GetGammaFromDeexcitation(int decayIndex);
  std::vector<G4PrimaryParticle*> GetGammasFromPositroniumAnnihilation(int decayIndex);

private:
  PositroniumDecayModelParams fModelParams;
  std::vector<GatePositronium> fPositroniumDecayChannel;
};

#endif
