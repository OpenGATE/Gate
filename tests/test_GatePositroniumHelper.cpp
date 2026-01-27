#include <cassert>
#include <iostream>

#include <TGeant4SystemOfUnits.h>

#include <GatePositronium.hh>
#include <GatePositroniumHelper.hh>
#include <GatePositroniumDecayModel.hh>

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
    bool res = true;
    std::unique_ptr<GateRunManager> runManager(new GateRunManager);
    initializeGateRunManager(runManager.get());

    GatePositroniumHelper helper;

    std::vector<float> fractions = {2, 12, 4, 2};
    std::vector<float> lifetimes = {0.125, 0.4, 2.0, 50.0};
    std::vector<PositronElectronInteraction> decays = {kParaPs, kDirect, kOrthoPs, kOrthoPs};
    std::vector<float> normFractions = helper.NormalizeFractions(fractions);
    std::vector<float> goodFractions = {0.1, 0.6, 0.2, 0.1};
    for (int i=0; i<normFractions.size(); i++) {
      if (normFractions.at(i) != goodFractions.at(i)) {
        res = false;
        std::cerr << "NormalizeFractions error" << std::endl;
      }
    }

    float intensityDirect2G = helper.CalcFractionFromDirectLifetime(goodFractions.at(1), PositroniumDecayKind::k2Gamma);
    float intensityDirect3G = helper.CalcFractionFromDirectLifetime(goodFractions.at(1), PositroniumDecayKind::k3Gamma);
    float goodIntensDirect3G = goodFractions.at(1)/372.;
    if (intensityDirect3G != goodIntensDirect3G || intensityDirect2G != goodFractions.at(1) - goodIntensDirect3G) {
      res = false;
      std::cerr << "Calculation of direct annihilation intensity error" << std::endl;
    }

    float intensityOPs2G = helper.CalcFractionFromOPsLifetime(goodFractions.at(2), lifetimes.at(2), PositroniumDecayKind::k2Gamma);
    float intensityOPs3G = helper.CalcFractionFromOPsLifetime(goodFractions.at(2), lifetimes.at(2), PositroniumDecayKind::k3Gamma);
    float goodIntensOPs3G = lifetimes.at(2)*goodFractions.at(2)/142.;
    if (intensityOPs3G != goodIntensOPs3G || intensityOPs2G != goodFractions.at(2) - goodIntensOPs3G) {
      res = false;
      std::cerr << "Calculation of o-Ps intensity error" << std::endl;
    }

    float intensityPps = helper.CalcPPsFractionFromOPs(goodFractions, decays);
    if (intensityPps != goodFractions.at(0)) {
      res = false;
      std::cerr << "Calculation of p-Ps intensity error" << std::endl;
    }

    return res;
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
