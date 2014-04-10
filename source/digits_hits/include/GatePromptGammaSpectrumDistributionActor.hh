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

//-----------------------------------------------------------------------------
class GatePromptGammaSpectrumDistributionActor : public GateVActor
{
public:
  virtual ~GatePromptGammaSpectrumDistributionActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaSpectrumDistributionActor)

  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*);
  virtual void BeginOfEventAction(const G4Event*) ;
  virtual void UserSteppingAction(const GateVVolume*, const G4Step*);

  virtual void PreUserTrackingAction(const GateVVolume*, const G4Track*);
  virtual void PostUserTrackingAction(const GateVVolume*, const G4Track*);
  virtual void EndOfEventAction(const G4Event*);

  virtual void SaveData();
  virtual void ResetData();

protected:
  GatePromptGammaSpectrumDistributionActor(G4String name, G4int depth=0);

};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(PromptGammaSpectrumDistributionActor,
                        GatePromptGammaSpectrumDistributionActor)

#endif // GATEPROMPTGAMMASPECTRUMDISTRIBUTIONACTOR
