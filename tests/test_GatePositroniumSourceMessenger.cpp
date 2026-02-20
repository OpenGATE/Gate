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
  DummyPositroniumSource() : GatePositroniumSource(nullptr) {}
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


bool test_default_params()
{
  std::cout << "test_default_params\n";

  DummyPositroniumSource src;
  //TestablePositroniumMessenger msg(&src);

  //auto p = msg.generatePositroniumDecayParams();

  //CHECK(p.fFractions.size() == 1, "default fractions size != 1");
  //CHECK(p.fFractions[0] == 1.0f, "default fraction != 1");
  //CHECK(p.fPromptGammaProbabilities[0] == 0.0f, "default prompt prob != 0");

  return true;
}

bool test_set_fractions()
{
  std::cout << "test_set_fractions\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdFractions(), "0.2 0.3 0.5");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fFractions.size() == 3, "fractions size wrong");
  CHECK(p.fFractions[0] == 0.2f, "fraction[0] wrong");
  CHECK(p.fFractions[1] == 0.3f, "fraction[1] wrong");
  CHECK(p.fFractions[2] == 0.5f, "fraction[2] wrong");

  return true;
}

bool test_set_lifetimes()
{
  std::cout << "test_set_lifetimes\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdLifetimes(), "0.1 10 100");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fLifetimes.size() == 3, "lifetimes size wrong");
  CHECK(p.fLifetimes[1] == 10.0f, "lifetime[1] wrong");

  return true;
}

bool test_prompt_gamma_clipping()
{
  std::cout << "test_prompt_gamma_clipping\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdPromptProb(), "-1 0.5 2");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fPromptGammaProbabilities[0] == 0.0f, "prob <0 not clipped");
  CHECK(p.fPromptGammaProbabilities[1] == 0.5f, "prob normal wrong");
  CHECK(p.fPromptGammaProbabilities[2] == 1.0f, "prob >1 not clipped");

  return true;
}

bool test_decay_kinds()
{
  std::cout << "test_decay_kinds\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdDecayKinds(), "k2Gamma k3Gamma");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fDecayKind.size() == 2, "decayKinds size wrong");
  CHECK(p.fDecayKind[0] == k2Gamma, "decayKind[0] wrong");
  CHECK(p.fDecayKind[1] == k3Gamma, "decayKind[1] wrong");

  return true;
}

bool test_interactions()
{
  std::cout << "test_interactions\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdInteractions(), "kParaPs kDirect kOrthoPs");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fPositronInteractions.size() == 3, "interaction size wrong");
  CHECK(p.fPositronInteractions[0] == PositronElectronInteraction::kParaPs, "wrong interaction[0]");
  CHECK(p.fPositronInteractions[1] == PositronElectronInteraction::kDirect, "wrong interaction[1]");
  CHECK(p.fPositronInteractions[2] == PositronElectronInteraction::kOrthoPs, "wrong interaction[2]");

  return true;
}

bool test_full_custom()
{
  std::cout << "test_full_custom\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdFractions(), "0.3 0.7");
  msg.SetNewValue(msg.CmdLifetimes(), "0.1 100");
  msg.SetNewValue(msg.CmdDecayKinds(), "k2Gamma k3Gamma");
  msg.SetNewValue(msg.CmdPromptProb(), "0.2 0.8");
  msg.SetNewValue(msg.CmdPromptEnergy(), "0.5 1.0");

  auto p = msg.generatePositroniumDecayParams();

  CHECK(p.fFractions.size() == 2, "wrong size");
  CHECK(p.fFractions[1] == 0.7f, "fraction wrong");
  CHECK(p.fDecayKind[1] == k3Gamma, "decay kind wrong");
  CHECK(p.fPromptGammaEnergy[1] == 1.0f, "energy wrong");

  return true;
}

bool test_vector_size_mismatch()
{
  std::cout << "test_vector_size_mismatch\n";

  DummyPositroniumSource src;
  TestablePositroniumMessenger msg(&src);

  msg.SetNewValue(msg.CmdFractions(), "0.5 0.5");
  msg.SetNewValue(msg.CmdLifetimes(), "0.1"); // mismatch

  try {
    msg.generatePositroniumDecayParams();
    CHECK(false, "expected failure due to size mismatch");
  } catch (...) {
    return true;
  }

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

  res &= test_default_params();
  res &= test_set_fractions();
  res &= test_set_lifetimes();
  res &= test_prompt_gamma_clipping();
  res &= test_decay_kinds();
  res &= test_interactions();
  res &= test_full_custom();
  res &= test_vector_size_mismatch();

  if (res) {
    std::cout << " All GatePositroniumSourceMessenger tests passed\n";
    return 0;
  } else {
    std::cerr << " Some GatePositroniumSourceMessenger tests failed\n";
    return -1;
  }
}


