
/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GatePositroniumDecayParamsGenerator_hh
#define GatePositroniumDecayParamsGenerator_hh

#include <optional>
#include <vector>

#include "GatePositroniumDecayModelParams.hh"

/*! class GatePositroniumDecayParamsGenerator
 *  brief Brief class description
 *
 *  Detailed description
 */
class GatePositroniumDecayParamsGenerator
{
public:
  GatePositroniumDecayParamsGenerator()= default;
  virtual ~GatePositroniumDecayParamsGenerator()=default;

  void SetEnableDeexcitation(const std::vector<bool>& IsPromptPhoton);
  void SetPromptGammaEnergies(const std::vector<float>& energies);
  void SetPostroniumLifetimes(const std::vector<float>& fPositroniumLifetimes);
  void SetDecayKinds(const std::vector<PositroniumDecayKind>& decayKinds);
  void SetPositroniumFraction(const std::vector<float>& positroniumFractions);

  PositroniumDecayModelParams generatePositroniumDecayParams() const;

private:
  std::optional<std::vector<float>> fPositroniumFractions;
  std::optional<std::vector<float>> fPositroniumLifetimes;
  std::optional<std::vector<bool>> fIsPromptPhoton;
  std::optional<std::vector<float>> fPromptPhotonEnergies;
  std::optional<std::vector<PositroniumDecayKind>> fDecayKinds;
};
#endif
