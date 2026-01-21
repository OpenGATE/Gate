#ifndef GatePositroniumHelper_h
#define GatePositroniumHelper_h 1

#include "GatePositroniumDecayModelParams.hh"

class GatePositroniumHelper
{
public:
  GatePositroniumHelper() {};
  virtual ~GatePositroniumHelper() {};

  struct PositroniumDecayModelParams CalculateFractionsFromLifetimes(struct PositroniumDecayModelParams params);
  float CalcPPsFractionFromOPs(std::vector<float> fractions, std::vector<PositronElectronInteraction> decays);
  std::pair<float, float> CalcFractionsFromLifetime(float intensity, float lifetime, PositronElectronInteraction inter);
  float CalcFractionFromOPsLifetime(float intensity, float lifetime, PositroniumDecayKind decay);
  float CalcFractionFromDirectLifetime(float intensity, PositroniumDecayKind decay);
  std::vector<float> NormalizeFractions (std::vector<float> fractions);

private:
//For now stored as a internal parameter. To check if there is already defined value somewhere
  static constexpr float fParaPsLifetime = 0.125; //ps
  static constexpr float fParaToOrthoPsFraction = 1.0/3.0;
  static constexpr float fOrthoPsMeanLifetime = 142.; //ns
  static constexpr float fHyperfineCoefficient = 372;
};

#endif
