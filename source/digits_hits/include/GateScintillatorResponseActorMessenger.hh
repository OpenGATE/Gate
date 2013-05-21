/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateScintillatorResponseActorMessenger
  \author simon.rit@creatis.insa-lyon.fr
*/

#ifndef GATESCINTILLATORRESPONSEACTORMESSENGER_HH
#define GATESCINTILLATORRESPONSEACTORMESSENGER_HH

#include "GateImageActorMessenger.hh"

class GateScintillatorResponseActor;
class GateScintillatorResponseActorMessenger : public GateImageActorMessenger
{
public:
  GateScintillatorResponseActorMessenger(GateScintillatorResponseActor* sensor);
  virtual ~GateScintillatorResponseActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateScintillatorResponseActor * pScintillatorResponseActor;
  G4UIcmdWithAString * pReadMuAbsortionListCmd;
  G4UIcmdWithABool * pEnableScatterCmd;
  G4UIcmdWithAString * pSetScatterOrderFilenameCmd;
};

#endif /* end #define GATESCINTILLATORRESPONSEACTORMESSENGER_HH*/
