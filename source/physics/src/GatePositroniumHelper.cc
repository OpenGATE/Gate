/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>

#include "GateMessageManager.hh"
#include "GatePositroniumHelper.hh"
#include "GatePositroniumConstants.hh"

using namespace GatePositroniumConstants;

PositroniumDecayModelParams GatePositroniumHelper::CalculateFractionsFromLifetimes(PositroniumDecayModelParams params)
{
  PositroniumDecayModelParams paramsOut;
  bool pPsExist = false;
  int pPsIndex = -1; 
  for (unsigned i=0; i<params.fPositronInteractions.size(); i++) {
    if (params.fPositronInteractions.at(i) == PositronElectronInteraction::kParaPs) {
      assert(pPsIndex<0); 
      pPsIndex = i; 
    } else {
      /// fraction passed to CalcFractionsFromLifetime is really an overall intensity here.
      std::pair<float, float> intens = CalcFractionsFromLifetime(params.fFractions.at(i), params.fLifetimes.at(i), params.fPositronInteractions.at(i));

      paramsOut.fFractions.push_back(intens.first);  //2G intens
      paramsOut.fFractions.push_back(intens.second); //3G intens
      paramsOut.fDecayKind.push_back(PositroniumDecayKind::k2Gamma);
      paramsOut.fDecayKind.push_back(PositroniumDecayKind::k3Gamma);

      paramsOut.fLifetimes.push_back(params.fLifetimes.at(i));
      paramsOut.fLifetimes.push_back(params.fLifetimes.at(i));
      paramsOut.fPromptGammaProbabilities.push_back(params.fPromptGammaProbabilities.at(i));
      paramsOut.fPromptGammaProbabilities.push_back(params.fPromptGammaProbabilities.at(i));
      paramsOut.fPromptGammaEnergy.push_back(params.fPromptGammaEnergy.at(i));
      paramsOut.fPromptGammaEnergy.push_back(params.fPromptGammaEnergy.at(i));
      paramsOut.fPositronInteractions.push_back(params.fPositronInteractions.at(i));
      paramsOut.fPositronInteractions.push_back(params.fPositronInteractions.at(i));

      if (params.fElectronCaptureProbabilities.size() > 0) {
        paramsOut.fElectronCaptureProbabilities.push_back(params.fElectronCaptureProbabilities.at(i));
        paramsOut.fElectronCaptureProbabilities.push_back(params.fElectronCaptureProbabilities.at(i));
      }
      if (params.fMeanPositronRange.size() > 0) {
        paramsOut.fMeanPositronRange.push_back(params.fMeanPositronRange.at(i));
        paramsOut.fMeanPositronRange.push_back(params.fMeanPositronRange.at(i));
      }
    }
  }
  if (!paramsOut.fPositronInteractions.size()) {
    GateError("GatePositroniumHelper::CalculateFractionsFromLifetimes: Could not calculate fractions from lifetimes. fPositronInteractions is empty.");
    return params;
  }

// setting pPs
  float pPsIntens = CalcPPsFractionFromOPs(paramsOut.fFractions, paramsOut.fPositronInteractions);
  if (pPsIntens > 0 && pPsIndex < 0) {
    GateError("GatePositroniumHelper::CalculateFractionsFromLifetimes: A non-zero para-Ps fraction was derived from the ortho-Ps components, but no kParaPs entry was provided in fPositronInteractions to supply its lifetime/prompt-gamma parameters.");
    return params;
  }
  if (pPsIntens > 0) {
    assert(pPsIndex >=0);
    paramsOut.fFractions.push_back(pPsIntens);
    paramsOut.fLifetimes.push_back(params.fLifetimes.at(pPsIndex));
    paramsOut.fPromptGammaProbabilities.push_back(params.fPromptGammaProbabilities.at(pPsIndex));
    paramsOut.fPromptGammaEnergy.push_back(params.fPromptGammaEnergy.at(pPsIndex));
    paramsOut.fDecayKind.push_back(PositroniumDecayKind::k2Gamma);
    paramsOut.fPositronInteractions.push_back(PositronElectronInteraction::kParaPs);
    if (params.fElectronCaptureProbabilities.size() > 0) {
      paramsOut.fElectronCaptureProbabilities.push_back(params.fElectronCaptureProbabilities.at(pPsIndex));
    }
    if (params.fMeanPositronRange.size() > 0) {
      paramsOut.fMeanPositronRange.push_back(params.fMeanPositronRange.at(pPsIndex));
    }
  }
  paramsOut.fFractions = NormalizeFractions(paramsOut.fFractions);
  return paramsOut;
}

float GatePositroniumHelper::CalcPPsFractionFromOPs(std::vector<float> fractions, std::vector<PositronElectronInteraction> decays) {
  float sum = 0.;
  auto itFrac = fractions.begin();
  auto itDec = decays.begin();
  while (itFrac != fractions.end() && itDec != decays.end()) {
    if (*itDec == PositronElectronInteraction::kOrthoPs)
      sum += *itFrac;
    ++itFrac;
    ++itDec;
  }
  return sum*kParaToOrthoPsFraction ;
}

std::pair<float, float> GatePositroniumHelper::CalcFractionsFromLifetime(float intensity, float lifetime, PositronElectronInteraction inter) {
  float intens2G = 1., intens3G = 0.;
  switch (inter) {
    case PositronElectronInteraction::kDirect:
      intens2G = intensity*(kHyperfineCoefficient  - 1.)/kHyperfineCoefficient;
      intens3G = intensity/kHyperfineCoefficient;
      break;
    case PositronElectronInteraction::kOrthoPs:
      intens2G = intensity*(kOrthoPsMeanLifetime_ns  - lifetime)/kOrthoPsMeanLifetime_ns;
      intens3G = intensity*lifetime/kOrthoPsMeanLifetime_ns;
      break;
    case PositronElectronInteraction::kParaPs:
      GateError("GatePositroniumHelper::CalcFractionsFromLifetimes: This function does not handle parapositrionium case. It should never be callded with inter == PositronElectronInteraction::kParaPs");
      break;
  }
  return std::make_pair(intens2G, intens3G);
}

std::vector<float> GatePositroniumHelper::NormalizeFractions(std::vector<float> fractions) {
  std::vector<float> normalizedVector = fractions;
  double sum = std::accumulate(fractions.begin(), fractions.end(), 0.0);
  if (sum > 0)
    std::transform(fractions.begin(), fractions.end(), normalizedVector.begin(), [sum](float element){ return element/sum; });
  else {
//ThrowException
    normalizedVector = fractions;
  }
  return normalizedVector;
}
