/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About class: Data structure holding all per-component parameters for the positronium decay model: fractions, lifetimes, decay kinds (2γ/3γ), positron-electron interaction types, electron capture probabilities, prompt gamma parameters, and positron range settings.
 **/

#ifndef GatePositroniumDecayModelParams_hh
#define GatePositroniumDecayModelParams_hh

#include <vector>

enum PositroniumDecayKind {k2Gamma, k3Gamma};

enum PositronElectronInteraction {kParaPs, kDirect, kOrthoPs};

struct PositroniumDecayModelParams
{
  std::vector<float> fFractions;
  std::vector<float> fLifetimes;
  std::vector<float> fElectronCaptureProbabilities;
  std::vector<float> fPromptGammaProbabilities;
  std::vector<float> fPromptGammaEnergy;
  std::vector<PositroniumDecayKind> fDecayKind;
  std::vector<PositronElectronInteraction> fPositronInteractions;
  std::vector<bool> fMeanPositronRangeEnabled;
  std::vector<float> fMeanPositronRange;
};

#endif
