
#include "GateMessageManager.hh" 
#include "GatePositroniumDecayParamsGenerator.hh"
#include <cassert>

void GatePositroniumDecayParamsGenerator::SetEnablePromptGamma(const std::vector<bool>& isPromptPhoton)
{
  fIsPromptPhoton = isPromptPhoton;
}
void GatePositroniumDecayParamsGenerator::SetDecayKinds(const std::vector<PositroniumDecayKind>& decayKinds)
{
  fDecayKinds = decayKinds;
}
void GatePositroniumDecayParamsGenerator::SetPositroniumLifetimes(const std::vector<float>& positroniumLifetimes)
{
  fPositroniumLifetimes = positroniumLifetimes;
}
void GatePositroniumDecayParamsGenerator::SetPromptGammaEnergies(const std::vector<float>& energies)
{
  fPromptPhotonEnergies = energies;
}
void GatePositroniumDecayParamsGenerator::SetPositroniumFraction(const std::vector<float>& positroniumFractions)
{
  fPositroniumFractions = positroniumFractions; 
}

PositroniumDecayModelParams GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams(const GatePositroniumDecayParamsGenerator::DecayModel model) const
{
  PositroniumDecayModelParams params;
  if (model == GatePositroniumDecayParamsGenerator::kParaPositronium) {
    params.fFractions= {1};
    params.fLifetimes= {0.1244}; // [ns]
    params.fDecayKind= {k2Gamma};
    if(fIsPromptPhoton.has_value() && fPromptPhotonEnergies.has_value()) {
      params.fIsPromptPhoton=fIsPromptPhoton.value();
      assert(params.fIsPromptPhoton.size()==1);
      params.fPromptPhotonEnergy=fPromptPhotonEnergies.value();
      assert(params.fPromptPhotonEnergy.size()==1);
    } else {
      params.fIsPromptPhoton={false};
      params.fPromptPhotonEnergy ={0.0};
    }
  }
  if (model == GatePositroniumDecayParamsGenerator::kOrthoPositronium) {
    params.fFractions= {1};
    params.fLifetimes= {142}; // [ns]
    params.fDecayKind= {k3Gamma};
    if(fIsPromptPhoton.has_value() && fPromptPhotonEnergies.has_value()) {
      params.fIsPromptPhoton=fIsPromptPhoton.value();
      assert(params.fIsPromptPhoton.size()==1);
      params.fPromptPhotonEnergy=fPromptPhotonEnergies.value();
      assert(params.fPromptPhotonEnergy.size()==1);
    } else {
      params.fIsPromptPhoton={false};
      params.fPromptPhotonEnergy ={0.0};
    }
  }

  if (model == GatePositroniumDecayParamsGenerator::kPositronium) {
    if(fPositroniumFractions.has_value()) {
      params.fFractions=fPositroniumFractions.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Positronium decay fractions are not set");
    }

    if(fPositroniumLifetimes.has_value()) {
      params.fLifetimes=fPositroniumLifetimes.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Positronium lifetimes are not set");
    }

    if(fDecayKinds.has_value()) {
      params.fDecayKind=fDecayKinds.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Positronium decay kinds are not set");
    }

    if(fIsPromptPhoton.has_value()) {
      params.fIsPromptPhoton=fIsPromptPhoton.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Positronium is prompt photon flags are not set");
    }

    if(fPromptPhotonEnergies.has_value()) {
      params.fPromptPhotonEnergy=fPromptPhotonEnergies.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Prompt photon energies are not set");
    }
  }

  return params;
}
