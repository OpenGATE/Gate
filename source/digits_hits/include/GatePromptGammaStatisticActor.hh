/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR_HH
#define GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR_HH

#include "GateConfiguration.h"
#include "GateVActor.hh"
#include "GatePromptGammaStatisticActorMessenger.hh"
#include "GatePromptGammaData.hh"
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//-----------------------------------------------------------------------------
class GatePromptGammaStatisticActor : public GateVActor
{
public:
  virtual ~GatePromptGammaStatisticActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaStatisticActor)

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
  GatePromptGammaStatisticActor(G4String name, G4int depth=0);
  GatePromptGammaStatisticActorMessenger * pMessenger;

  GatePromptGammaData data;

  bool sigma_filled;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaStatisticActor,
                        GatePromptGammaStatisticActor)

#endif // GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR
