/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEDOSESOURCEACTORMESSENGER_HH
#define GATEDOSESOURCEACTORMESSENGER_HH

#include "GateConfiguration.h"
#include "GateImageActorMessenger.hh"

class GateDoseSourceActor;

//-----------------------------------------------------------------------------
class GateDoseSourceActorMessenger: public GateImageActorMessenger
{
public:

  GateDoseSourceActorMessenger(GateDoseSourceActor*);
  ~GateDoseSourceActorMessenger();
  void SetNewValue(G4UIcommand*, G4String);

protected:
  void BuildCommands(G4String base);
  GateDoseSourceActor* pDoseSourceActor;

  G4UIcmdWithAString* bSpotIDFromSourceCmd;
  G4UIcmdWithAString* bLayerIDFromSourceCmd;
};
//-----------------------------------------------------------------------------

#endif // End GATEDOSESOURCEACTORMESSENGER
