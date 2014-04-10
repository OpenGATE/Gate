/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH
#define GATEPROMPTGAMMAPRODUCTIONTLEACTOR_HH

#include "GateConfiguration.h"
#include "GateVActor.hh"
#include "GateActorMessenger.hh"
#include "GatePromptGammaProductionTLEActorMessenger.hh"

//-----------------------------------------------------------------------------
class GatePromptGammaProductionTLEActor: public GateVActor
{
public:
  virtual ~GatePromptGammaProductionTLEActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GatePromptGammaProductionTLEActor)

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
  GatePromptGammaProductionTLEActor(G4String name, G4int depth=0);

};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(GpTLEActor,GatePromptGammaProductionTLEActor)

#endif // end GATEPROMPTGAMMAPRODUCTIONTLEACTOR
