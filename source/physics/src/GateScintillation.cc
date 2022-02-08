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
    G4MaterialPropertyVector* Intensity1 =
        aMaterialPropertiesTable->GetProperty("SCINTILLATIONCOMPONENT1");
    G4MaterialPropertyVector* Intensity2 =
        aMaterialPropertiesTable->GetProperty("SCINTILLATIONCOMPONENT2");
    G4MaterialPropertyVector* Intensity3 =
        aMaterialPropertiesTable->GetProperty("SCINTILLATIONCOMPONENT3");

    this->SetFiniteRiseTime(false);

    if(Intensity1 && aMaterialPropertiesTable->ConstPropertyExists("SCINTILLATIONRISETIME1"))
      this->SetFiniteRiseTime(true);

    if(Intensity2 && aMaterialPropertiesTable->ConstPropertyExists("SCINTILLATIONRISETIME2"))
      this->SetFiniteRiseTime(true);

    if(Intensity3 && aMaterialPropertiesTable->ConstPropertyExists("SCINTILLATIONRISETIME3"))
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
