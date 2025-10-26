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

//bool run_tests2()
//{
  //std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  //initializeGateRunManager(runManager.get());

  //PositroniumDecayModelParams params;
  //params.fFractions={0.3,0.7};
  //params.fLifetimes={0.1244 ,138.6};
  //params.fDecayKind={PositroniumDecayKind::k2Gamma, PositroniumDecayKind::k3Gamma};
  //MiniPositroniumDecayModel model(params);

  //return true;
//}

bool run_tests()
{
  bool res = true;
  GatePositronium pPs("pPs", 0.1 , 2); 
  if (pPs.GetLifeTime() != 0.1) {
    return false;
  }
  if (pPs.GetName() != "pPs") {
    return false;
  }
  if (pPs.GetAnnihilationGammasNumber() != 2) {
    return false;
  }
  return res;
}

bool run_tests2()
{
  bool res = true;
  GatePositronium oPs("oPs", 1000, 3); 
  if (oPs.GetLifeTime() != 1000) {
    return false;
  }
  if (oPs.GetName() != "oPs") {
    return false;
  }
  if (oPs.GetAnnihilationGammasNumber() != 3) {
    return false;
  }
  return res;
}


int main()
{
  // Initialization
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

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
