// unitCuTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <thread>
int test_sumReduction(int argc, char** argv);

int main(int argc, char** argv)
{
  int x = 0;

  x = test_sumReduction(argc, argv);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  return x;
}
