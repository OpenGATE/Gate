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
#include "GateActorMessenger.hh"

#include <G4UIcmdWithAnInteger.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>

class GatePromptGammaProductionTLEActor;

//-----------------------------------------------------------------------------
class GatePromptGammaProductionTLEActorMessenger: public GateActorMessenger
{
public:

  GatePromptGammaProductionTLEActorMessenger(GatePromptGammaProductionTLEActor*);
  ~GatePromptGammaProductionTLEActorMessenger();
  void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);
  GatePromptGammaProductionTLEActor* pTLEActor;

};
//-----------------------------------------------------------------------------

#endif // End GATEPROMPTGAMMAPRODUCTIONTLEACTOR
