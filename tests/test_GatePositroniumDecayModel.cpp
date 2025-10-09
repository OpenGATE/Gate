#include <cassert>
#include <iostream>
#include <GatePositroniumDecayModel.hh>

bool run_tests()
{
  PositroniumDecayModelParams params;
  MiniPositroniumDecayModel model(params);
  auto res = model.getPositroniumDecayParamsForEvent();
  return true;
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
