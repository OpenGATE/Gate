/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifndef GATESCATTERORDERTRACKINFORMATIONACTORMESSENGER_HH
#define GATESCATTERORDERTRACKINFORMATIONACTORMESSENGER_HH

#include "globals.hh"
#include "GateScatterOrderTrackInformationActor.hh"
#include "GateActorMessenger.hh"
#include "GateUIcmdWith2Vector.hh"

class GateScatterOrderTrackInformationActor;
class GateScatterOrderTrackInformationActorMessenger: public GateActorMessenger
{
public:
  GateScatterOrderTrackInformationActorMessenger(GateScatterOrderTrackInformationActor* sensor);
  virtual ~GateScatterOrderTrackInformationActorMessenger();

  //NOTE: we keep the messenger member functions just in case we want to add new options to the actor
  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateScatterOrderTrackInformationActor * pScatterOrderActor;
};

#endif /* end #define GATESCATTERORDERTRACKINFORMATIONACTORMESSENGER_HH*/
