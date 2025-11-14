#include <cassert>

#include "GateMessageManager.hh"
#include "GatePositroniumDecayParamsGenerator.hh"

void GatePositroniumDecayParamsGenerator::SetEnablePromptGamma(const std::vector<bool>& isPromptGamma)
{
  fIsPromptGamma = isPromptGamma;
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
  fPromptGammaEnergies = energies;
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
    if(fIsPromptGamma.has_value() && fPromptGammaEnergies.has_value()) {
      params.fIsPromptGamma=fIsPromptGamma.value();
      assert(params.fIsPromptGamma.size()==1);
      params.fPromptGammaEnergy=fPromptGammaEnergies.value();
      assert(params.fPromptGammaEnergy.size()==1);
    } else {
      params.fIsPromptGamma={false};
      params.fPromptGammaEnergy ={0.0};
    }
  }
  if (model == GatePositroniumDecayParamsGenerator::kOrthoPositronium) {
    params.fFractions= {1};
    params.fLifetimes= {142}; // [ns]
    params.fDecayKind= {k3Gamma};
    if(fIsPromptGamma.has_value() && fPromptGammaEnergies.has_value()) {
      params.fIsPromptGamma=fIsPromptGamma.value();
      assert(params.fIsPromptGamma.size()==1);
      params.fPromptGammaEnergy=fPromptGammaEnergies.value();
      assert(params.fPromptGammaEnergy.size()==1);
    } else {
      params.fIsPromptGamma={false};
      params.fPromptGammaEnergy ={0.0};
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

    if(fIsPromptGamma.has_value()) {
      params.fIsPromptGamma=fIsPromptGamma.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Positronium is prompt photon flags are not set");
    }

    if(fPromptGammaEnergies.has_value()) {
      params.fPromptGammaEnergy=fPromptGammaEnergies.value();
    } else {
      GateError("GateExtendedVSource::generatePositroniumDecayParams: Prompt photon energies are not set");
    }
  }

  return params;
}
