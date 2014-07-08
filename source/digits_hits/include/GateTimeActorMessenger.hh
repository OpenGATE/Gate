/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATETIMEACTORMESSENGER_HH
#define GATETIMEACTORMESSENGER_HH

#include "GateVActor.hh"
#include "GateTimeActor.hh"
#include "GateActorMessenger.hh"
#include "G4UIcmdWithABool.hh"

class GateTimeActor;

class GateTimeActorMessenger : public GateActorMessenger
{
public:
  GateTimeActorMessenger(GateTimeActor* sensor);
  virtual ~GateTimeActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateTimeActor * pTimeActor;
  G4UIcmdWithABool * pEnableDetailedStatCmd;

};

#endif /* end #define GATETIMEACTORMESSENGER_HH*/
