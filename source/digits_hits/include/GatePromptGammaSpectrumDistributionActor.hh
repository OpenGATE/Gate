/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR_HH
#define GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR_HH

#include "GateConfiguration.h"
#include "GateVActor.hh"
#include "GatePromptGammaSpectrumDistributionActorMessenger.hh"
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaSpectrumDistributionActor : public GateVActor
{
public:
  virtual ~GatePromptGammaSpectrumDistributionActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaSpectrumDistributionActor)

  virtual void Construct();
  virtual void UserSteppingAction(const GateVVolume*, const G4Step*);
  virtual void SaveData();
  virtual void ResetData();

  void SetProtonEMin(G4double x);
  void SetProtonEMax(G4double x);
  void SetGammaEMin(G4double x);
  void SetGammaEMax(G4double x);
  void SetProtonNbBins(G4int x);
  void SetGammaNbBins(G4int x);

protected:
  GatePromptGammaSpectrumDistributionActor(G4String name, G4int depth=0);
  GatePromptGammaSpectrumDistributionActorMessenger * pMessenger;

  // Histograms limits
  double min_proton_energy;
  double min_gamma_energy;
  double max_proton_energy;
  double max_gamma_energy;
  int proton_bin;
  int gamma_bin;

  // Data
  TFile* pTfile;
  TH2D* pHEpEpg;
  TH2D* pHEpEpgNormalized;
  TH1D* pHEpInelastic;
  TH1D* pHEp;
  TH1D* pHEpInelasticProducedGamma;
  TH1D* pHEpSigmaInelastic;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaSpectrumDistributionActor,
                        GatePromptGammaSpectrumDistributionActor)

#endif // GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR
