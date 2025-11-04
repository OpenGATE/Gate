#include <cassert>
#include <iostream>
#include <memory>

#include "GatePositroniumDecayParamsGenerator.hh"
#include "GatePositroniumDecayModelParams.hh"
#include "GateRunManager.hh"
#include "GatePhysicsList.hh"
#include "GateDetectorConstruction.hh"

#define CHECK(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr << "Test failure: " << msg \
                << " (in " << __FUNCTION__ << ", line " << __LINE__ << ")\n"; \
      return false; \
    } \
  } while(0)

void initializeGateRunManager(GateRunManager* runManager)
{
  GateDetectorConstruction* gateDC = new GateDetectorConstruction();
  runManager->SetUserInitialization(gateDC);
  runManager->SetUserInitialization(GatePhysicsList::GetInstance());
  runManager->InitializeAll();
}

bool test_para_default()
{
  std::cout << "test_para_default" << std::endl;
  GatePositroniumDecayParamsGenerator gen;
  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kParaPositronium);

  CHECK(p.fFractions[0] == 1, "para fraction != 1");
  CHECK(p.fLifetimes[0] == 0.1244f, "para lifetime wrong");
  CHECK(p.fDecayKind[0] == k2Gamma, "para decay kind wrong");
  CHECK(p.fIsPromptPhoton[0] == false, "para prompt flag wrong");
  CHECK(p.fPromptPhotonEnergy[0] == 0.0f, "para prompt energy wrong");
  return true;
}

bool test_ortho_default()
{
  std::cout << "test_ortho_default" << std::endl;
  GatePositroniumDecayParamsGenerator gen;
  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kOrthoPositronium);

  CHECK(p.fFractions[0] == 1, "ortho fraction != 1");
  CHECK(p.fLifetimes[0] == 138.6f, "ortho lifetime wrong");
  CHECK(p.fDecayKind[0] == k3Gamma, "ortho decay kind wrong");
  CHECK(p.fIsPromptPhoton[0] == false, "ortho prompt flag wrong");
  CHECK(p.fPromptPhotonEnergy[0] == 0.0f, "ortho prompt energy wrong");
  return true;
}

bool test_para_prompt_gamma()
{
  std::cout << "test_para_prompt_gamma" << std::endl;
  GatePositroniumDecayParamsGenerator gen;
  gen.SetEnableDeexcitation({true});
  gen.SetPromptGammaEnergies({0.511f});

  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kParaPositronium);

  CHECK(p.fIsPromptPhoton[0] == true, "para prompt photon flag wrong");
  CHECK(p.fPromptPhotonEnergy[0] == 0.511f, "para prompt photon energy wrong");
  return true;
}

bool test_positronium_custom()
{
  std::cout << "test_positronium_custom" << std::endl;
  GatePositroniumDecayParamsGenerator gen;

  gen.SetPositroniumFraction({0.3f, 0.7f});
  gen.SetPostroniumLifetimes({0.12f, 140.0f});
  gen.SetDecayKinds({k2Gamma, k3Gamma});
  gen.SetEnableDeexcitation({false, true});
  gen.SetPromptGammaEnergies({0.0f, 0.511f});

  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kPositronium);

  CHECK(p.fFractions[1] == 0.7f, "PS fraction mismatch");
  CHECK(p.fLifetimes[1] == 140.0f, "PS lifetime mismatch");
  CHECK(p.fDecayKind[1] == k3Gamma, "PS decay kind mismatch");
  CHECK(p.fIsPromptPhoton[1] == true, "PS prompt flag mismatch");
  CHECK(p.fPromptPhotonEnergy[1] == 0.511f, "PS prompt energy mismatch");
  return true;
}

bool test_missing_params_should_fail()
{
  std::cout << "test_missing_params_should_fail" << std::endl;
  GatePositroniumDecayParamsGenerator gen;

  bool caught = false;
  try {
    auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kPositronium);
  } catch (...) {
    caught = true;
  }
  CHECK(caught, "Missing positronium params did NOT throw");
  return true;
}

bool test_vector_size_mismatch()
{
  std::cout << "test_vector_size_mismatch" << std::endl;
  GatePositroniumDecayParamsGenerator gen;

  gen.SetPositroniumFraction({0.5f, 0.5f});
  gen.SetPostroniumLifetimes({0.12f}); // mismatch!
  gen.SetDecayKinds({k2Gamma, k3Gamma});
  gen.SetEnableDeexcitation({false, false});
  gen.SetPromptGammaEnergies({0.0f, 0.0f});

  bool caught = false;
  try {
    auto p = gen.generatePositroniumDecayParams();
  } catch (...) {
    caught = true;
  }
  CHECK(caught, "Mismatch in vector sizes did NOT throw");
  return true;
}

int main()
{
  std::unique_ptr<GateRunManager> runManager(new GateRunManager);
  initializeGateRunManager(runManager.get());

  bool res = true;
  res &= test_para_default();
  res &= test_ortho_default();
  res &= test_para_prompt_gamma();
  res &= test_positronium_custom();
  res &= test_missing_params_should_fail();
  res &= test_vector_size_mismatch();

  if (res) {
    std::cout << " All PositroniumDecayParamsGenerator tests passed\n";
    return 0;
  } else {
    std::cerr << " Some PositroniumDecayParamsGenerator tests failed\n";
    return -1;
  }
}

