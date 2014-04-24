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

class GatePromptGammaProductionTLEActor;

//-----------------------------------------------------------------------------
class GatePromptGammaProductionTLEActorMessenger: public GateImageActorMessenger
{
public:

  GatePromptGammaProductionTLEActorMessenger(GatePromptGammaProductionTLEActor*);
  ~GatePromptGammaProductionTLEActorMessenger();
  void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);
  GatePromptGammaProductionTLEActor* pTLEActor;

  G4UIcmdWithAString * pSetInputDataFileCmd;
};
//-----------------------------------------------------------------------------

#endif // End GATEPROMPTGAMMAPRODUCTIONTLEACTOR
