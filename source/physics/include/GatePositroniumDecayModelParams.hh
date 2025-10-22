/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositroniumDecayModelParams_hh
#define GatePositroniumDecayModelParams_hh

#include <vector>

enum PositroniumDecayKind { k2Gamma, k3Gamma};

struct PositroniumDecayModelParams
{
  std::vector<float> fFractions;
  std::vector<float> fLifetimes;
  std::vector<bool> fIsPromptPhoton;
  std::vector<float> fPromptPhotonEnergy;
  std::vector<PositroniumDecayKind> fDecayKind;
};

#endif
