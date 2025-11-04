#include <cassert>
#include <iostream>

#include <TGeant4SystemOfUnits.h> 

#include <GatePositronium.hh>

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

bool test_pPs_properties()
{
  std::cout << "test_pPs_properties\n";
  GatePositronium pPs("pPs", 0.1 * ns);

  CHECK(pPs.GetName() == "pPs", "Name mismatch");
  CHECK(std::abs(pPs.GetLifeTime() - 0.1 * ns) < 1e-12, "Lifetime mismatch");
  CHECK(pPs.GetAnnihilationGammasNumber() == 2, "pPs should decay to 2 gammas");

  return true;
}

bool test_oPs_properties()
{
  std::cout << "test_oPs_properties\n";
  GatePositronium oPs("oPs", 142.0 * ns);

  CHECK(oPs.GetName() == "oPs", "Name mismatch");
  CHECK(oPs.GetAnnihilationGammasNumber() == 3, "oPs should decay to 3 gammas");

  return true;
}

bool test_move_semantics()
{
  std::cout << "test_move_semantics\n";

  GatePositronium oPs("oPs", 1000);
  GatePositronium moved = std::move(oPs);

  CHECK(moved.GetName() == "oPs", "move: name lost");
  CHECK(moved.GetAnnihilationGammasNumber() == 3, "move: wrong gamma count");

  return true;
}


/// todo: add test to check what happens if pPs with 3 ?
bool test_in_vector()
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

bool test_decay_products()
{
  std::cout << "test_decay_products\n";
  GatePositronium pPs("pPs", 0.1 * ns);

  auto* products = pPs.GetDecayProducts();
  CHECK(products != nullptr, "DecayProducts should not be null");
  CHECK(products->entries() == 2, "pPs should produce 2 daughters");

  return true;
}


int main()
{
  // Initialization
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

  bool ok = true;
  ok = ok & test_pPs_properties();
  ok = ok & test_oPs_properties();
  ok = ok & test_move_semantics();
  ok = ok & test_in_vector();
  ok = ok & test_decay_products();

  if (ok) {
    std::cout << "All tests have passed" << std::endl;
    return 0;
  } else {
    std::cerr << "Some tests failed" << std::endl;
    return -1;
  }
}
