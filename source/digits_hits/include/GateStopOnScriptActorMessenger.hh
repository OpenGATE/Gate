/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateStopOnScriptActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATESTOPONSCRIPTACTORMESSENGER_HH
#define GATESTOPONSCRIPTACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateActorMessenger.hh"

class GateStopOnScriptActor;
class GateStopOnScriptActorMessenger : public GateActorMessenger
{
public:
  GateStopOnScriptActorMessenger(GateStopOnScriptActor* sensor);
  virtual ~GateStopOnScriptActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateStopOnScriptActor * pActor;
  G4UIcmdWithABool * pEnableSaveAllActorsCmd;
};

#endif /* end #define GATESTOPONSCRIPTACTORMESSENGER_HH*/
