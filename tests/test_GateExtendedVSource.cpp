#include <cassert>
#include <iostream>

#include <TGeant4SystemOfUnits.h> 

#include <GateExtendedVSource.hh>

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
  GateExtendedVSource source("source1");
  if(source.GetName() !="source1") {
    return false;
  }
  source.SetType("mPs");
  if(source.GetType() !="mPs") {
    return false;
  }

  G4Event* pEvent = nullptr;
  source.GeneratePrimaries(pEvent);
  return true;
}


int main()
{
  // Initialization
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

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
