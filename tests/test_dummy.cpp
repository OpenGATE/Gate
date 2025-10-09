#include <cassert>
#include <iostream>
bool run_tests()
{
  if (1==0) {
    return true;
  }
  return false;
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
