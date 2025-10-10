#include <cassert>
#include <iostream>
#include <vector>
#include <GatePositroniumDecayModel.hh>

/// It should be separated in several subtests
bool run_tests()
{
  bool res = true;
  PositroniumDecayModelParams params;
  MiniPositroniumDecayModel model(params);
  int index = MiniPositroniumDecayModel::getPositroniumDecayIndex({1});
  if (index != 0) {
    res = false;
    std::cerr << "getPositroniumDecayDecayIndex({1})!=0" << std::endl;
  }
  index = MiniPositroniumDecayModel::getPositroniumDecayIndex({});
  if (index != -1) {
    res = false;
    std::cerr << "getPositroniumDecayDecayIndex({})!=-1" << std::endl;
  }

  float epsilon = 1e-3;
  int num_of_trials = 1e6;
  std::vector<float> fractions={0.5,0.4,0.1};
  std::vector<int> indices = {0,0,0};
  for (int i = 0; i < num_of_trials; i++) {
    index = MiniPositroniumDecayModel::getPositroniumDecayIndex(fractions);
    indices[index] = indices[index] +1;
  }
  std::vector<float> estimated_fractions = {0.,0., 0.};
  for (int i = 0; i < estimated_fractions.size(); i++) {
    estimated_fractions[i] = float(indices[i])/num_of_trials;
    std::cout << "estimated_fractions[i]="<<estimated_fractions[i]<< std::endl;
  }

  for (int i = 0; i < estimated_fractions.size(); i++) {
    if(std::abs(fractions[i] - estimated_fractions[i])>epsilon)
    {
      res = false;
      std::cerr << "assumed fractions and estimated fractions differ: std::abs(fractions[i] - estimated_fractions[i])" << std::endl;
    }
  }
  return res;
}

int main()
{
  bool res = true;
  res = res & run_tests();
  if (res) {
    std::cout << "All tests have passed" << std::endl;
    return 0;
  } else {
    std::cerr << "Some tests failed" << std::endl;
    return -1;
  }
}
