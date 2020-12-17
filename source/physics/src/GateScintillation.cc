#include "GateScintillation.hh"

//----------------------------------------------------------------------------------------
G4VParticleChange*
GateScintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
{
  const G4Material* aMaterial = aTrack.GetMaterial();
  G4MaterialPropertiesTable* aMaterialPropertiesTable =
      aMaterial->GetMaterialPropertiesTable();

  if(aMaterialPropertiesTable)
  {
    G4MaterialPropertyVector* Fast_Intensity =
        aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
    G4MaterialPropertyVector* Slow_Intensity =
        aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");

    this->SetFiniteRiseTime(false);

    if(Fast_Intensity && aMaterialPropertiesTable->ConstPropertyExists("FASTSCINTILLATIONRISETIME"))
      this->SetFiniteRiseTime(true);

    if(Slow_Intensity && aMaterialPropertiesTable->ConstPropertyExists("SLOWSCINTILLATIONRISETIME"))
      this->SetFiniteRiseTime(true);

  }


  return G4Scintillation::PostStepDoIt(aTrack, aStep);
}

G4bool GateScintillation::IsApplicable(const G4ParticleDefinition &aParticleType)
{
  if (aParticleType.GetParticleName() == "opticalphoton") return false;
  if (aParticleType.IsShortLived()) return false;

  return true;
}
//----------------------------------------------------------------------------------------
