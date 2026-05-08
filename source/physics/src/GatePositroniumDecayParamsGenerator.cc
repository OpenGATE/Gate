/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <cassert>

#include "GateMessageManager.hh"
#include "GatePositroniumHelper.hh"
#include "GatePositroniumDecayParamsGenerator.hh"

void GatePositroniumDecayParamsGenerator::SetElectronCaptureProbabilities(const std::vector<float>& electronCaptureProb)
{
  fElectronCaptureProbabilities = electronCaptureProb;
}

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

void GatePositroniumDecayParamsGenerator::SetMeanPositronRange(const std::vector<float>& meanPositironRange)
{
  fMeanPositronRange = meanPositironRange;
}


PositroniumDecayModelParams GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams() const
{
  PositroniumDecayModelParams params;
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

  auto nElements = fPositroniumFractions.value().size();
  if(fElectronCaptureProbabilities.has_value()) {
    params.fElectronCaptureProbabilities=fElectronCaptureProbabilities.value();
  } else {
    params.fElectronCaptureProbabilities.assign(nElements, 0.0);
  }

  if(fDecayKinds.has_value()) {
    params.fDecayKind=fDecayKinds.value();
  } else if (!params.fPositronInteractions.empty() && !params.fLifetimes.empty() && !params.fFractions.empty()) {
    params = positronHelper.CalculateFractionsFromLifetimes(params);
  } else {
    GateError("GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: Positronium decay kinds are not set and one or more sets that can calculate them (positronInteractions, lifetimes or fractions) is/are empty");
  }

  if(fMeanPositronRange.has_value()) {
    params.fMeanPositronRangeEnabled.assign(nElements, true);
    params.fMeanPositronRange=fMeanPositronRange.value();
  } else {
    params.fMeanPositronRangeEnabled.assign(nElements, false);
    params.fMeanPositronRange.assign(nElements, 0.0);
  }

validatePositroniumDecayParams(params);

return params;
}

void GatePositroniumDecayParamsGenerator::validatePositroniumDecayParams(const PositroniumDecayModelParams& params) const
{
  auto ref_param_number = params.fDecayKind.size();
  bool size_mismatch = (ref_param_number != params.fFractions.size()) ||
                       (ref_param_number != params.fPromptGammaProbabilities.size()) ||
                       (ref_param_number != params.fLifetimes.size()) ||
                       (ref_param_number != params.fPromptGammaEnergy.size()) ||
                       (ref_param_number != params.fElectronCaptureProbabilities.size());
  if (size_mismatch) {
    std::cout << ref_param_number << " " << params.fFractions.size() << " " << params.fPromptGammaProbabilities.size() << " " << params.fLifetimes.size() << " " << params.fPromptGammaEnergy.size() << params.fElectronCaptureProbabilities.size() << std::endl;
    GateError(
        "GatePositroniumDecayParamsGenerator::generatePositroniumDecayParams: "
        "number of provided parameters in Fractions, PromptGamma, Lifetimes, "
        "Gamma Energies, Electron Capture Probabilites are not the same");
  }
  constexpr double kProb_of_no_particle_limit = 0.9;
  for (int i = 0; i < ref_param_number; i++) {
    auto electron_capture_prob = params.fElectronCaptureProbabilities[i];
    auto no_prompt_prob = 1 -params.fPromptGammaProbabilities[i];
    if (electron_capture_prob *no_prompt_prob >= kProb_of_no_particle_limit) {
      GateWarning("The probability of 0 prompt emission times probability of electron capture (no anihillation) is larger than 90%. Please check channel definitions! Otherwise simulations can take a lot of time");
    }
  }
}
