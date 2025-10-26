/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateMiniPositroniumDecayModel_hh
#define GateMiniPositroniumDecayModel_hh

#include<vector>

#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"

#include "GateGammaEmissionModel.hh"
#include "GatePositroniumDecayModelParams.hh"
#include "GatePositronium.hh"

// Todo change the name to GatePositroniumDecayModel
// Todo2: add docs
class MiniPositroniumDecayModel:public GateGammaEmissionModel
{
  public:
  static int getPositroniumDecayIndex(const std::vector<float>& fractions); 

  public:
  explicit MiniPositroniumDecayModel(const PositroniumDecayModelParams& modelParams);

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
