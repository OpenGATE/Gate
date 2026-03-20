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
  enum DecayModel {kParaPositronium, kOrthoPositronium, kPositronium};

  GatePositroniumDecayParamsGenerator()=default;
  virtual ~GatePositroniumDecayParamsGenerator()=default;

  void SetPromptGammaProbabilities(const std::vector<float>& promptGammaProb);
  void SetPromptGammaEnergies(const std::vector<float>& energies);
  void SetPositroniumLifetimes(const std::vector<float>& fPositroniumLifetimes);
  void SetDecayKinds(const std::vector<PositroniumDecayKind>& decayKinds);
  void SetPositronInteractions(const std::vector<PositronElectronInteraction>& positronInteractions);
  void SetPositroniumFraction(const std::vector<float>& positroniumFractions);
  void SetMeanPositronRange(const std::vector<float>& meanPositronRange);

  PositroniumDecayModelParams generatePositroniumDecayParams(DecayModel model=kPositronium) const;

private:
  std::optional<std::vector<float>> fPositroniumFractions;
  std::optional<std::vector<float>> fPositroniumLifetimes;
  std::optional<std::vector<float>> fPromptGammaProbabilities;
  std::optional<std::vector<float>> fPromptGammaEnergies;
  std::optional<std::vector<PositroniumDecayKind>> fDecayKinds;
  std::optional<std::vector<PositronElectronInteraction>> fPositronInteractions;
  std::optional<std::vector<bool>> fMeanPositronRangeEnabled;
  std::optional<std::vector<float>> fMeanPositronRange;
};
#endif
