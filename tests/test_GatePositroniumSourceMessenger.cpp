#include <iostream>
#include <memory>

#include "GatePositroniumSourceMessenger.hh"
#include "GatePositroniumSource.hh"
#include "GateRunManager.hh"
#include "GateDetectorConstruction.hh"
#include "GatePhysicsList.hh"

#include "TestingTools.h"

class DummyPositroniumSource : public GatePositroniumSource {
public:
  DummyPositroniumSource() : GatePositroniumSource("Ps") {}
};

class TestablePositroniumMessenger : public GatePositroniumSourceMessenger {
public:
  using GatePositroniumSourceMessenger::GatePositroniumSourceMessenger;

  G4UIcommand* CmdFractions()     { return upCmdSetPositroniumFractions.get(); }
  G4UIcommand* CmdLifetimes()     { return upCmdSetPositroniumLifetimes.get(); }
  G4UIcommand* CmdDecayKinds()    { return upCmdSetDecayKinds.get(); }
  G4UIcommand* CmdPromptProb()    { return upCmdSetPromptPhotonProbabilites.get(); }
  G4UIcommand* CmdPromptEnergy()  { return upCmdSetPromptPhotonEnergies.get(); }
  G4UIcommand* CmdInteractions()  { return upCmdSetPositronInteractions.get(); }
};

void initializeGateRunManager(GateRunManager* runManager)
{
  GateDetectorConstruction* gateDC = new GateDetectorConstruction();
  runManager->SetUserInitialization(gateDC);
  runManager->SetUserInitialization(GatePhysicsList::GetInstance());
  runManager->InitializeAll();
}

bool test_full_custom()
{
  std::cout << "test_full_custom\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdFractions(), "0.3 0.7");
  msg.SetNewValue(msg.CmdLifetimes(), "0.1 100 ns");
  msg.SetNewValue(msg.CmdDecayKinds(), "k2Gamma k3Gamma");
  msg.SetNewValue(msg.CmdPromptProb(), "0.2 0.8");
  msg.SetNewValue(msg.CmdPromptEnergy(), "0.5 1.0 MeV");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fFractions.size() == 2, "wrong size");
  CHECK(p.fFractions[1] == 0.7f, "fraction wrong");
  CHECK(p.fDecayKind[1] == k3Gamma, "decay kind wrong");
  CHECK(p.fPromptGammaEnergy[1] == 1.0f, "energy wrong");

  return true;
}


bool test_unit_handling_in_commands()
{
  std::cout << "test_unit_handling_in_commands\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdFractions(), "0.3 0.7");
  msg.SetNewValue(msg.CmdLifetimes(), "0.1 100 ns");
  msg.SetNewValue(msg.CmdDecayKinds(), "k2Gamma k3Gamma");
  msg.SetNewValue(msg.CmdPromptProb(), "0.2 0.8");
  msg.SetNewValue(msg.CmdPromptEnergy(), "0.5 1.0 keV");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fFractions.size() == 2, "wrong size");
  CHECK(p.fFractions[1] == 0.7f, "fraction wrong");
  CHECK(p.fDecayKind[1] == k3Gamma, "decay kind wrong");
  CHECK(p.fPromptGammaEnergy[1] == 1.0f /1000, "energy wrong"); // Cause we used keV and MeV is the default unit

  return true;
}
// ==========================================================
// main()
// ==========================================================
int main()
{
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

  bool res = true;

  res &= test_full_custom();
  res &= test_unit_handling_in_commands();

  if (res) {
    std::cout << " All GatePositroniumSourceMessenger tests passed\n";
    return 0;
  } else {
    std::cerr << " Some GatePositroniumSourceMessenger tests failed\n";
    return -1;
  }
}


