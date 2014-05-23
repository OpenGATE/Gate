/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
  \class  GateActorManagerMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEACTORMANAGERMESSENGER_HH
#define GATEACTORMANAGERMESSENGER_HH

#include "globals.hh"

#include "G4UImessenger.hh"

#include "GateUIcmdWith2String.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"


class GateActorManager;
class G4UIdirectory;

class GateActorManagerMessenger : public  G4UImessenger
{
public:
  GateActorManagerMessenger(GateActorManager* sMan);
  virtual ~GateActorManagerMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateActorManager * pActorManager;

  GateUIcmdWith2String * pAddActor;
  G4UIcmdWithoutParameter * pInitActor;
  G4UIcmdWithABool *pResetAfterSaving;

  G4UIdirectory*            pActorCommand;
};

#endif /* end #define GATEACTORMANAGERMESSENGER_HH */
