#ifndef GatePositroniumHelper_h
#define GatePositroniumHelper_h 1

#include "GatePositroniumDecayModelParams.hh"
#include "GatePositroniumDecayModel.hh"
#include "globals.hh"

enum PositronElectronInteraction {kpPs, kDirect, koPs};

class GatePositroniumHelper
{
public:
  GatePositroniumHelper() {};
  virtual ~GatePositroniumHelper() {};

  PositroniumDecayModelParams CalculateFractionsFromLifetimes(PositroniumDecayModelParams params);
  float CalcPPsFractionFromOPs(std::vector<float> fractions, std::vector<PositronElectronInteraction> decays);
  std::pair<float, float> CalcFractionsFromLifetime(float intensity, float lifetime, PositronElectronInteraction inter);
  float CalcFractionFromOPsLifetime(float intensity, float lifetime, PositroniumDecayKind decay);
  float CalcFractionFromDirectLifetime(float intensity, PositroniumDecayKind decay);
  std::vector<float> NormalizeFractions (std::vector<float> fractions);

private:
//For now stored as a internal parameter. To check if there is already defined value somewhere
  float fpPsLifetime = 0.125; //ps
  float fPPsToOPsFrac = 1.0/3.0;
  float fOPsMeanLifetime = 142.; //ns
  float fHyperfineCoef = 372;
};

#endif
