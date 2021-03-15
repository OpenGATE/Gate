/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATESIMULATIOSTATISTICACTORMESSENGER_HH
#define GATESIMULATIOSTATISTICACTORMESSENGER_HH

#include "GateConfiguration.h"
#include "GateActorMessenger.hh"
#include <G4UIcmdWithABool.hh>

class GateSimulationStatisticActor;

//-----------------------------------------------------------------------------
class GateSimulationStatisticActorMessenger : public GateActorMessenger {
public:

    GateSimulationStatisticActorMessenger(GateSimulationStatisticActor *);

    ~GateSimulationStatisticActorMessenger();

    void SetNewValue(G4UIcommand *, G4String);

protected:
    void BuildCommands(G4String base);

    GateSimulationStatisticActor *pActor;
    G4UIcmdWithABool *pTrackTypesFlagCmd;
};
//-----------------------------------------------------------------------------

#endif
