/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPROMPTGAMMAPRODUCTIONTLEACTORMESSENGER_HH
#define GATEPROMPTGAMMAPRODUCTIONTLEACTORMESSENGER_HH

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
  GatePromptGammaAnalogActor* pTLEActor;

  G4UIcmdWithAString * pSetInputDataFileCmd;
  G4UIcmdWithABool * pEnableUncertaintyCmd;
  G4UIcmdWithABool * pEnableIntermediaryUncertaintyOutputCmd;
};
//-----------------------------------------------------------------------------

#endif // End GATEPROMPTGAMMAPRODUCTIONTLEACTOR
