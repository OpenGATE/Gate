/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATESINGLEFIXEDFORCEDDECTECTIONACTORMESSENGER_HH
#define GATESINGLEFIXEDFORCEDDECTECTIONACTORMESSENGER_HH

#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"

#include "globals.hh"
#include "GateFixedForcedDetectionActorMessenger.hh"
#include "GateSingleFixedForcedDetectionActor.hh"
#include "GateActorMessenger.hh"
#include "GateUIcmdWith2Vector.hh"

class GateSingleFixedForcedDetectionActor;
class GateSingleFixedForcedDetectionActorMessenger: public GateFixedForcedDetectionActorMessenger
{
public:
  GateSingleFixedForcedDetectionActorMessenger(GateSingleFixedForcedDetectionActor* sensor);
  virtual ~GateSingleFixedForcedDetectionActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateSingleFixedForcedDetectionActor * pActor;
  G4UIcmdWithAString * pSetSingleInteractionFilenameCmd;
  G4UIcmdWithAString * pSetSingleInteractionTypeCmd;
  G4UIcmdWith3Vector * pSetSingleInteractionDirectionCmd;
  G4UIcmdWith3VectorAndUnit * pSetSingleInteractionPositionCmd;
  G4UIcmdWithADoubleAndUnit * pSetSingleInteractionEnergyCmd;
  G4UIcmdWithAnInteger * pSetSingleInteractionZCmd;
};

#endif /* end #define GATESINGLEFIXEDFORCEDDECTECTIONACTORMESSENGER_HH*/
#endif // GATE_USE_RTK
