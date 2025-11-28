#include <algorithm>
#include <numeric>
#include <vector>

#include "GateMessageManager.hh"
#include "GatePositroniumHelper.hh"
#include "globals.hh"

PositroniumDecayModelParams GatePositroniumHelper::CalculateFractionsFromLifetimes(PositroniumDecayModelParams params)
{
  PositroniumDecayModelParams paramsOut;
  for (unsigned i=0; i<params.fPositronInteractions.size(); i++) {
    if (params.fPositronInteractions.at(i) != PositronElectronInteraction::kpPs) {
      std::pair<float, float> intens = CalcFractionsFromLifetime(params.fFractions.at(i), params.fLifetimes.at(i), params.fPositronInteractions.at(i));
      paramsOut.fFractions.push_back(intens.first);  //2G intens
      paramsOut.fFractions.push_back(intens.second); //3G intens
      paramsOut.fLifetimes.push_back(params.fLifetimes.at(i));
      paramsOut.fLifetimes.push_back(params.fLifetimes.at(i));
      paramsOut.fPromptGammaProbabilities.push_back(params.fPromptGammaProbabilities.at(i));
      paramsOut.fPromptGammaProbabilities.push_back(params.fPromptGammaProbabilities.at(i));
      paramsOut.fPromptGammaEnergy.push_back(params.fPromptGammaEnergy.at(i));
      paramsOut.fPromptGammaEnergy.push_back(params.fPromptGammaEnergy.at(i));
      paramsOut.fDecayKind.push_back(PositroniumDecayKind::k2Gamma);
      paramsOut.fDecayKind.push_back(PositroniumDecayKind::k3Gamma);
      paramsOut.fPositronInteractions.push_back(params.fPositronInteractions.at(i));
      paramsOut.fPositronInteractions.push_back(params.fPositronInteractions.at(i));
    }
  }
  if (!paramsOut.fPositronInteractions.size()) {
    GateError("GatePositroniumHelper::CalculateFractionsFromLifetimes: Could not calculate fraction from lifetimes");
    return params;
  }
// setting pPs
  float pPsIntens = CalcPPsFractionFromOPs(paramsOut.fFractions, paramsOut.fPositronInteractions)
  if (pPsIntens > 0) {
    paramsOut.fFractions.push_back(pPsIntens);
    paramsOut.fLifetimes.push_back(fpPsLifetime);
    paramsOut.fPromptGammaProbabilities.push_back(paramsOut.fPromptGammaProbabilities.at(0));
    paramsOut.fPromptGammaEnergy.push_back(paramsOut.fPromptGammaEnergy.at(0));
    paramsOut.fDecayKind.push_back(PositroniumDecayKind::k2Gamma);
    paramsOut.fPositronInteractions.push_back(PositronElectronInteraction::koPs);
  }

  paramsOut.fFractions = NormalizeFractions(paramsOut.fFractions);
  return paramsOut;
}

float GatePositroniumHelper::CalcPPsFractionFromOPs(std::vector<float> fractions, std::vector<PositronElectronInteraction> decays) {
  float sum = 0.;
  auto itFrac = fractions.begin();
  auto itDec = decays.begin();
  while (itFrac != fractions.end() && itDec != decays.end()) {
    if (*itDec == PositronElectronInteraction::koPs)
      sum += *itFrac;
    ++itFrac;
    ++itDec;
  }
  return sum*fPPsToOPsFrac;
}

std::pair<float, float> GatePositroniumHelper::CalcFractionsFromLifetime(float intensity, float lifetime, PositronElectronInteraction inter) {
  float intens2G = 1., intens3G = 0.;
  switch (inter) {
    case PositronElectronInteraction::kDirect:
      intens2G = intensity*(fHyperfineCoef - 1.)/fHyperfineCoef;
      intens3G = intensity/fHyperfineCoef;
      break;
    case PositronElectronInteraction::koPs:
      intens2G = intensity*(fOPsMeanLifetime - lifetime)/fOPsMeanLifetime;
      intens3G = intensity*lifetime/fOPsMeanLifetime;
      break;
  }
  return std::make_pair(intens2G, intens3G);
}

float GatePositroniumHelper::CalcFractionFromOPsLifetime(float intensity, float lifetime, PositroniumDecayKind decay) {
  float nominator = 0.;
  if (intensity > 0 && lifetime > 0) {
    switch (decay) {
      case PositroniumDecayKind::k2Gamma:
        nominator = fOPsMeanLifetime - lifetime;
        break;
      case PositroniumDecayKind::k3Gamma:
        nominator = lifetime;
        break;
// If there will be more decays one needs to modify it
    }
    return intensity*nominator/fOPsMeanLifetime;
  } else
    return nominator;
}

float GatePositroniumHelper::CalcFractionFromDirectLifetime(float intensity, PositroniumDecayKind decay) {
  float nominator = 0.;
  if (intensity > 0) {
    switch (decay) {
      case PositroniumDecayKind::k2Gamma:
        nominator = fHyperfineCoef - 1.;
        break;
      case PositroniumDecayKind::k3Gamma:
        nominator = 1.;
        break;
    }
    return intensity*nominator/fHyperfineCoef;
  } else
    return nominator;
}

std::vector<float> GatePositroniumHelper::NormalizeFractions(std::vector<float> fractions) {
  std::vector<float> normalizedVector;
  double sum = std::accumulate(fractions.begin(), fractions.end(), 0.0);
  if (sum > 0)
    std::transform(fractions.begin(), fractions.end(), normalizedVector.begin(), [sum](float element){ return element/sum; });
  else {
//ThrowException
    normalizedVector = fractions;
  }
  return normalizedVector;
}
