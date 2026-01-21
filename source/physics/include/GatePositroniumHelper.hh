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
};

#endif
