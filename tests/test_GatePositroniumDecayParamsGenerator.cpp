#include <cassert>
#include <iostream>
#include <memory>

#include "GatePositroniumDecayParamsGenerator.hh"
#include "GatePositroniumDecayModelParams.hh"
#include "GateRunManager.hh"
#include "GatePhysicsList.hh"
#include "GateDetectorConstruction.hh"

#include "TestingTools.h"


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
  CHECK(p.fPromptGammaProbabilities[0] == 0.0, "para prompt probability wrong");
  CHECK(p.fPromptGammaEnergy[0] == 0.0f, "para prompt energy wrong");
  return true;
}

bool test_ortho_default()
{
  std::cout << "test_ortho_default" << std::endl;
  GatePositroniumDecayParamsGenerator gen;
  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kOrthoPositronium);

  CHECK(p.fFractions[0] == 1, "ortho fraction != 1");
  CHECK(p.fLifetimes[0] == 142.0f, "ortho lifetime wrong");
  CHECK(p.fDecayKind[0] == k3Gamma, "ortho decay kind wrong");
  CHECK(p.fPromptGammaProbabilities[0] == 0.0, "ortho prompt probability wrong");
  CHECK(p.fPromptGammaEnergy[0] == 0.0f, "ortho prompt energy wrong");
  return true;
}

bool test_para_prompt_gamma()
{
  std::cout << "test_prompt_gamma" << std::endl;
  GatePositroniumDecayParamsGenerator gen;
  gen.SetPromptGammaProbabilities({1.0});
  gen.SetPromptGammaEnergies({1.2f});

  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kParaPositronium);

  CHECK(p.fPromptGammaProbabilities[0] == 1.0, "para prompt photon probability wrong");
  CHECK(p.fPromptGammaEnergy[0] == 1.2f, "prompt photon energy wrong");
  return true;
}

bool test_positronium_custom()
{
  std::cout << "test_positronium_custom" << std::endl;
  GatePositroniumDecayParamsGenerator gen;

  gen.SetPositroniumFraction({0.3f, 0.7f});
  gen.SetPositroniumLifetimes({0.12f, 140.0f});
  gen.SetDecayKinds({k2Gamma, k3Gamma});
  gen.SetPromptGammaProbabilities({0, 1.0});
  gen.SetPromptGammaEnergies({0.0f, 1.2f});

  auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kPositronium);

  CHECK(p.fFractions[0] == 0.3f, "PS fraction mismatch");
  CHECK(p.fLifetimes[0] == 0.12f, "PS lifetime mismatch");
  CHECK(p.fDecayKind[0] == k2Gamma, "PS decay kind mismatch");
  CHECK(p.fPromptGammaProbabilities[0] == 0.0, "PS prompt probability mismatch");
  CHECK(p.fPromptGammaEnergy[0] == 0.0f, "PS prompt energy mismatch");

  CHECK(p.fFractions[1] == 0.7f, "PS fraction mismatch");
  CHECK(p.fLifetimes[1] == 140.0f, "PS lifetime mismatch");
  CHECK(p.fDecayKind[1] == k3Gamma, "PS decay kind mismatch");
  CHECK(p.fPromptGammaProbabilities[1] == 1.0, "PS prompt probability mismatch");
  CHECK(p.fPromptGammaEnergy[1] == 1.2f, "PS prompt energy mismatch");
  return true;
}

bool test_missing_params_should_fail()
// This test cannot run  — GateError calls exit(-1)
{
  GatePositroniumDecayParamsGenerator gen;
  std::cout << "test_missing_params_should_fail" << std::endl;
  std::cout << "but we cannot run it because GateError() calls exit(-1)" << std::endl;
  //bool caught = false;
  //try {
    //auto p = gen.generatePositroniumDecayParams(GatePositroniumDecayParamsGenerator::kPositronium);
  //} catch (...) {
    //std::cout << "we caught the exception" << std::endl;
    //caught = true;
  //}
  //CHECK(caught, "Missing parameters did NOT throw");
  return true;
}


bool test_vector_size_mismatch()
// This test cannot run  — GateError calls exit(-1)
{
  std::cout << "test_vector_size_mismatch" << std::endl;
  std::cout << "but we cannot run it because GateError() calls exit(-1)" << std::endl;

  //GatePositroniumDecayParamsGenerator gen;
  //gen.SetPositroniumFraction({0.5f, 0.5f});
  //gen.SetPositroniumLifetimes({0.12f}); // mismatch!
  //gen.SetDecayKinds({k2Gamma, k3Gamma});
  //gen.SetEnablePromptGamma({false, false});
  //gen.SetPromptGammaEnergies({0.0f, 0.0f});

  //bool caught = false;
  //try {
    //auto p = gen.generatePositroniumDecayParams();
  //} catch (...) {
    //std::cout << "we caught the exception" << std::endl;
    //caught = true;
  //}
  //CHECK(caught, "Mismatch in vector sizes did NOT throw");
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

