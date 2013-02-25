/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_HH
#define GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_HH

#include "globals.hh"
#include "GateHybridForcedDetectionActor.hh"
#include "GateActorMessenger.hh"
#include "GateUIcmdWith2Vector.hh"

class GateHybridForcedDetectionActor;
class GateHybridForcedDetectionActorMessenger: public GateActorMessenger
{
public:
  GateHybridForcedDetectionActorMessenger(GateHybridForcedDetectionActor* sensor);
  virtual ~GateHybridForcedDetectionActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateHybridForcedDetectionActor * pHybridActor;
  G4UIcmdWithAString * pSetDetectorCmd;
  GateUIcmdWith2Vector * pSetDetectorResolCmd;
  G4UIcmdWithAString * pSetGeometryFilenameCmd;
  G4UIcmdWithAString * pSetPrimaryFilenameCmd;
  G4UIcmdWithAString * pSetMaterialMuFilenameCmd;
};

#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTORMESSENGER_HH*/
#endif // GATE_USE_RTK
