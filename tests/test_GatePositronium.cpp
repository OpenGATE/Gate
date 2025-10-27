#include <cassert>
#include <iostream>

#include <TGeant4SystemOfUnits.h> 

#include <GatePositronium.hh>

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

bool run_tests()
{
  GatePositronium pPs("pPs", 0.1); 
  if (pPs.GetLifeTime() != 0.1) {
    return false;
  }
  if (pPs.GetName() != "pPs") {
    return false;
  }
  if (pPs.GetAnnihilationGammasNumber() != 2) {
    return false;
  }
  return true;
}

bool run_tests2()
{
  GatePositronium oPs("oPs", 1000); 
  if (oPs.GetLifeTime() != 1000) {
    return false;
  }
  if (oPs.GetName() != "oPs") {
    return false;
  }
  if (oPs.GetAnnihilationGammasNumber() != 3) {
    return false;
  }
  return true;
}

/// todo: add test to check what happens if pPs with 3 ?
bool run_tests3()
{
  std::vector<GatePositronium> vect;
  vect.push_back(std::move(GatePositronium("oPs", 1000)));
  vect.push_back(std::move(GatePositronium("pPs", 0.1)));
  vect.push_back(std::move(GatePositronium("oPs", 2000)));
  vect.push_back(std::move(GatePositronium("oPs", 5000)));
  if(vect[0].GetName()!="oPs") {
    return false;
  }
  if(vect[1].GetName()!="pPs") {
    return false;
  }
  if(vect[2].GetName()!="oPs") {
    return false;
  }
  if(vect[3].GetName()!="oPs") {
    return false;
  }

  if(vect[0].GetLifeTime()!= 1000) {
    return false;
  }
  if(vect[1].GetLifeTime()!= 0.1) {
    return false;
  }
  if(vect[2].GetLifeTime()!=2000) {
    return false;
  }
  if(vect[3].GetLifeTime()!=5000) {
    return false;
  }

  if(vect[0].GetAnnihilationGammasNumber()!= 3) {
    return false;
  }
  if(vect[1].GetAnnihilationGammasNumber()!= 2) {
    return false;
  }
  if(vect[2].GetAnnihilationGammasNumber()!= 3) {
    return false;
  }
  if(vect[3].GetAnnihilationGammasNumber()!= 3) {
    return false;
  }

  return true;
}


int main()
{
  // Initialization
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

  bool res = true;
  res = res & run_tests();
  res = res & run_tests2();
  res = res & run_tests3();

  if (res) {
    std::cout << "All tests have passed" << std::endl;
    return 0;
  } else {
    std::cerr << "Some tests failed" << std::endl;
    return -1;
  }
}
