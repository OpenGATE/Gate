#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

#include <GatePositroniumDecayModel.hh>

#include "GateRunManager.hh"
#include "GatePhysicsList.hh"
#include "GateDetectorConstruction.hh"

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

int main()
{
  bool res = true;
  res = res & run_tests();
  res = res & run_tests2();

  if (res) {
    std::cout << "All tests have passed" << std::endl;
    return 0;
  } else {
    std::cerr << "Some tests failed" << std::endl;
    return -1;
  }
}
