/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <numeric>

#include <GatePositroniumDecayModel.hh>

#include "GateRunManager.hh"
#include "GatePhysicsList.hh"
#include "GateDetectorConstruction.hh"

#include "TestingTools.h"

void initializeGateRunManager(GateRunManager* runManager)
{
  // Set the DetectorConstruction
  GateDetectorConstruction* gateDC = new GateDetectorConstruction();
  runManager->SetUserInitialization( gateDC );
  // Set the PhysicsList
  runManager->SetUserInitialization( GatePhysicsList::GetInstance() );
  // Initialize G4 kernel
  runManager->InitializeAll();
}

bool run_tests2()
{
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

  PositroniumDecayModelParams params;
  params.fFractions={0.3,0.7};
  params.fLifetimes={0.1244 ,138.6};
  params.fDecayKind={PositroniumDecayKind::k2Gamma, PositroniumDecayKind::k3Gamma};
  GatePositroniumDecayModel model(params);

  return true;
}

/// It should be separated in several subtests
bool run_tests()
{
  bool res = true;
  PositroniumDecayModelParams params;
  GatePositroniumDecayModel model(params);
  int index = GatePositroniumDecayModel::getPositroniumDecayIndex({1});
  if (index != 0) {
    res = false;
    std::cerr << "getPositroniumDecayDecayIndex({1})!=0" << std::endl;
  }
  index = GatePositroniumDecayModel::getPositroniumDecayIndex({});
  if (index != -1) {
    res = false;
    std::cerr << "getPositroniumDecayDecayIndex({})!=-1" << std::endl;
  }

  float epsilon = 1e-3;
  int num_of_trials = 1e6;
  std::vector<float> fractions={0.5,0.4,0.1};
  std::vector<int> indices = {0,0,0};
  for (int i = 0; i < num_of_trials; i++) {
    index = GatePositroniumDecayModel::getPositroniumDecayIndex(fractions);
    indices[index] = indices[index] +1;
  }
  std::vector<float> estimated_fractions = {0.,0., 0.};
  for (int i = 0; i < estimated_fractions.size(); i++) {
    estimated_fractions[i] = float(indices[i])/num_of_trials;
  }

  for (int i = 0; i < estimated_fractions.size(); i++) {
    if(std::abs(fractions[i] - estimated_fractions[i])>epsilon)
    {
      res = false;
      std::cerr << "assumed fractions and estimated fractions differ:" << std::abs(fractions[i] - estimated_fractions[i]) << std::endl;
    }
  }
  return res;
}

bool run_tests_positron_range() {
  bool res = true;
  G4ThreeVector vec(0, 0, 0);
  G4double expectedRange = 0.1 * mm;
  double epsilon = 3e-3 * mm;
  G4int ntrials = 1000000;
  std::vector<G4ThreeVector> shiftedPos;
  std::vector<G4double> shiftedRadii(ntrials);
  for (int i = 0; i < ntrials; i++) {
    shiftedPos.push_back(
        GatePositroniumDecayModel::AddPositronRangeShift(vec, expectedRange));
  }
  std::transform(shiftedPos.begin(), shiftedPos.end(), shiftedRadii.begin(),
                 [](const G4ThreeVector &vec) { return vec.mag(); });
  auto radiiSum =
      std::accumulate(shiftedRadii.begin(), shiftedRadii.end(), 0.0);
  auto radiusMean = radiiSum / ntrials;
  if (std::abs(radiusMean - expectedRange) > epsilon) {
    res = false;
    std::cerr << "assumed range and estimated range differ, expected:"
              << expectedRange << ", determined:" << radiusMean << std::endl;
  }
  return res;
}

// Bug #2: getPositroniumDecayIndex returns -1 when fractions do not sum to 1.0.
// The caller (GeneratePrimaryVertices) uses the return value directly as a vector
// index on fElectronCaptureProbabilities, fMeanPositronRangeEnabled,
// fMeanPositronRange, and fPositroniumDecayChannel (lines 55, 76, 96, 112 of
// GatePositroniumDecayModel.cc), so -1 causes undefined behaviour.

bool test_decay_index_in_bounds_for_sub_unity_fractions()
{
  // Fractions {0.3, 0.3} sum to 0.6: roughly 40 % of calls return -1.
  // The test checks that the return value is always a valid index;
  // it fails reliably under the current implementation.
  const std::vector<float> fractions = {0.3f, 0.3f};
  for (int i = 0; i < 10000; ++i) {
    int index = GatePositroniumDecayModel::getPositroniumDecayIndex(fractions);
    CHECK(index >= 0 && index < static_cast<int>(fractions.size()),
          "getPositroniumDecayIndex returned " + std::to_string(index)
          + " for a vector of size " + std::to_string(fractions.size())
          + " — out-of-bounds index causes UB in GeneratePrimaryVertices (Bug #2)");
  }
  return true;
}

bool test_decay_index_float_fractions_sum_below_unity()
{
  // Even fractions that appear normalized can fail: in IEEE 754 single precision
  // (1.0f/3.0f)*3 = 1 - 2^-24 < 1.0, leaving a gap that G4UniformRand (a double)
  // can land in, returning -1 from getPositroniumDecayIndex.
  // This test confirms the precondition: the cumulative float sum of three equal
  // thirds is strictly below 1.0, so the gap exists.
  const std::vector<float> fractions = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
  float cumulative = 0.0f;
  for (float f : fractions) cumulative += f;
  CHECK(cumulative == 1.0f,
        "Float fractions do not sum to exactly 1.0 (actual sum = "
        + std::to_string(cumulative)
        + "). Any G4UniformRand() value in (sum, 1.0) causes getPositroniumDecayIndex"
        " to return -1 instead of the last valid index (Bug #2)");
  return true;
}

int main()
{
  bool res = true;
  res = res & run_tests();
  res = res & run_tests2();
  res = res & run_tests_positron_range();
  res = res & test_decay_index_in_bounds_for_sub_unity_fractions();
  res = res & test_decay_index_float_fractions_sum_below_unity();

  if (res) {
    std::cout << "All tests have passed" << std::endl;
    return 0;
  } else {
    std::cerr << "Some tests failed" << std::endl;
    return -1;
  }
}
