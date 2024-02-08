/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONTLEACTORMESSENGER_HH
#define GATEPROMPTGAMMAPRODUCTIONTLEACTORMESSENGER_HH

#include "GateConfiguration.h"
#include "GateImageActorMessenger.hh"

class GatePromptGammaTLEActor;

//-----------------------------------------------------------------------------
class GatePromptGammaTLEActorMessenger: public GateImageActorMessenger
{
public:

  GatePromptGammaTLEActorMessenger(GatePromptGammaTLEActor*);
  ~GatePromptGammaTLEActorMessenger();
  void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);
  GatePromptGammaTLEActor* pPGTLEActor;

  G4UIcmdWithAString * pSetInputDataFileCmd;
  G4UIcmdWithABool * pEnableDebugOutputCmd;
  G4UIcmdWithABool * pEnableOutputMatchCmd;
  G4UIcmdWithAnInteger * pTimeNbBinsCmd;
};
//-----------------------------------------------------------------------------

#endif // End GATEPROMPTGAMMAPRODUCTIONTLEACTOR
