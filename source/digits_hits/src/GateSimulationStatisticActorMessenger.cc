/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateSimulationStatisticActorMessenger.hh"
#include "GateSimulationStatisticActor.hh"

//-----------------------------------------------------------------------------
GateSimulationStatisticActorMessenger::GateSimulationStatisticActorMessenger(GateSimulationStatisticActor *v)
    : GateActorMessenger(v), pActor(v) {
    BuildCommands(baseName + pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSimulationStatisticActorMessenger::~GateSimulationStatisticActorMessenger() {
    delete pTrackTypesFlagCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSimulationStatisticActorMessenger::BuildCommands(G4String base) {
    G4String bb = base + "/setTrackTypesFlag";
    pTrackTypesFlagCmd = new G4UIcmdWithABool(bb, this);
    G4String guidance = G4String("If true, compute the nb of track per types");
    pTrackTypesFlagCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSimulationStatisticActorMessenger::SetNewValue(G4UIcommand *cmd,
                                                        G4String newValue) {
    if (cmd == pTrackTypesFlagCmd) pActor->SetTrackTypesFlag(pTrackTypesFlagCmd->GetNewBoolValue(newValue));
    GateActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------
