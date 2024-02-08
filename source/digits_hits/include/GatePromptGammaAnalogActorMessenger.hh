/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAANALOGACTORMESSENGER_HH
#define GATEPROMPTGAMMAANALOGACTORMESSENGER_HH

#include "GateConfiguration.h"
#include "GateImageActorMessenger.hh"

class GatePromptGammaAnalogActor;

//-----------------------------------------------------------------------------
class GatePromptGammaAnalogActorMessenger: public GateImageActorMessenger
{
public:

  GatePromptGammaAnalogActorMessenger(GatePromptGammaAnalogActor*);
  ~GatePromptGammaAnalogActorMessenger();
  void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);
  GatePromptGammaAnalogActor* pPGAnalogActor;

  G4UIcmdWithAString * pSetInputDataFileCmd;
  G4UIcmdWithABool * pSetOutputCountCmd;
  G4UIcmdWithAnInteger * pTimeNbBinsCmd;
};
//-----------------------------------------------------------------------------

#endif // End GATEPROMPTGAMMAANALOGACTOR
