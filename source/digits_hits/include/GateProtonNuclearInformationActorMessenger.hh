/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifndef GATEPROTONNUCLEARINFORMATIONACTORMESSENGER_HH
#define GATEPROTONNUCLEARINFORMATIONACTORMESSENGER_HH

#include "globals.hh"
#include "GateProtonNuclearInformationActor.hh"
#include "GateActorMessenger.hh"
#include "GateUIcmdWith2Vector.hh"

class GateProtonNuclearInformationActor;
class GateProtonNuclearInformationActorMessenger: public GateActorMessenger
{
public:
  GateProtonNuclearInformationActorMessenger(GateProtonNuclearInformationActor* sensor);
  virtual ~GateProtonNuclearInformationActorMessenger();

  //NOTE: we keep the messenger member functions just in case we want to add new options to the actor
  void BuildCommands(G4String);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateProtonNuclearInformationActor * pProtonNuclearInformationActor;
};

#endif /* end #define GATEPROTONNUCLEARINFORMATIONACTORMESSENGER_HH*/
