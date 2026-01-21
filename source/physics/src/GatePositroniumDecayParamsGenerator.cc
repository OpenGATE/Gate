#include <cassert>

#include "GateMessageManager.hh"
#include "GatePositroniumHelper.hh"
#include "GatePositroniumDecayParamsGenerator.hh"
#include "GatePositroniumConstants.hh"

using namespace gate_positronium_constants;

void GatePositroniumDecayParamsGenerator::SetPromptGammaProbabilities(const std::vector<float>& promptGammaProb)
{
  fPromptGammaProbabilities = promptGammaProb;
}
void GatePositroniumDecayParamsGenerator::SetDecayKinds(const std::vector<PositroniumDecayKind>& decayKinds)
{
  fDecayKinds = decayKinds;
}
void GatePositroniumDecayParamsGenerator::SetPositronInteractions(const std::vector<PositronElectronInteraction>& positronInteractions)
{
  fPositronInteractions = positronInteractions;
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
    params.fLifetimes= {kParaPsLifetime_ns}; // [ns]
    params.fDecayKind= {k2Gamma};
    params.fPositronInteractions= {kpPs};
    if(fPromptGammaProbabilities.has_value() && fPromptGammaEnergies.has_value()) {
      params.fPromptGammaProbabilities=fPromptGammaProbabilities.value();
      assert(params.fPromptGammaProbabilities.size()==1);
      params.fPromptGammaEnergy=fPromptGammaEnergies.value();
      assert(params.fPromptGammaEnergy.size()==1);
    } else {
      params.fPromptGammaProbabilities={0.0};
      params.fPromptGammaEnergy ={0.0};
    }
  }
  if (model == GatePositroniumDecayParamsGenerator::kOrthoPositronium) {
    params.fFractions= {1};
    params.fLifetimes= {kOrthoPsMeanLifetime_ns}; // [ns]
    params.fDecayKind= {k3Gamma};
    params.fPositronInteractions= {koPs};
    if(fPromptGammaProbabilities.has_value() && fPromptGammaEnergies.has_value()) {
      params.fPromptGammaProbabilities=fPromptGammaProbabilities.value();
      assert(params.fPromptGammaProbabilities.size()==1);
      params.fPromptGammaEnergy=fPromptGammaEnergies.value();
      assert(params.fPromptGammaEnergy.size()==1);
    } else {
      params.fPromptGammaProbabilities={0.0};
      params.fPromptGammaEnergy ={0.0};
    }
  }

  if (model == GatePositroniumDecayParamsGenerator::kPositronium) {
    GatePositroniumHelper positronHelper;
    if(fPositroniumFractions.has_value()) {
      params.fFractions=positronHelper.NormalizeFractions(fPositroniumFractions.value());
    } else {
      GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Positronium decay fractions are not set");
    }

    if(fPositroniumLifetimes.has_value()) {
      params.fLifetimes=fPositroniumLifetimes.value();
    } else {
      GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Positronium lifetimes are not set");
    }

    if(fPositronInteractions.has_value()) {
      params.fPositronInteractions=fPositronInteractions.value();
    } else if (!fDecayKinds.has_value()) {
      GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Positronium interactions are not set and decay kinds are also empty");
    }

    if(fPromptGammaProbabilities.has_value()) {
      params.fPromptGammaProbabilities=fPromptGammaProbabilities.value();
    } else {
      GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Positronium prompt gamma probabilities are not set");
    }

    if(fPromptGammaEnergies.has_value()) {
      params.fPromptGammaEnergy=fPromptGammaEnergies.value();
    } else {
      GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Prompt gamma energies are not set");
    }

    if(fDecayKinds.has_value()) {
      params.fDecayKind=fDecayKinds.value();
    } else if (!params.fPositronInteractions.empty() && !params.fLifetimes.empty() && !params.fFractions.empty()) {
      params = positronHelper.CalculateFractionsFromLifetimes(params);
    } else {
      GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Positronium decay kinds are not set and one or more sets that can calculate them (positronInteractions, lifetimes or fractions) is/are empty");
    }
  }

  auto ref_param_number = params.fDecayKind.size();
  bool size_mismatch = (ref_param_number != params.fFractions.size()) ||
                       (ref_param_number != params.fPromptGammaProbabilities.size()) ||
                       (ref_param_number != params.fLifetimes.size()) ||
                       (ref_param_number != params.fPromptGammaEnergy.size());
  if (size_mismatch) {
    std::cout << ref_param_number << " " << params.fFractions.size() << " " << params.fPromptGammaProbabilities.size() << " " << params.fLifetimes.size() << " " << params.fPromptGammaEnergy.size() << std::endl;
    GateError(
        "GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: "
        "number of provided parameters in Fractions, PromptGamma, Lifetimes, "
        "Gamma Energies are not the same");
  }

  return params;
}
