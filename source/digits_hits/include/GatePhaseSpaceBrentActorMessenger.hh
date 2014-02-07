/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GatePhaseSpaceBrentActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GatePhaseSpaceBrentACTORMESSENGER_HH
#define GatePhaseSpaceBrentACTORMESSENGER_HH

#include "GatePhaseSpaceActorMessenger.hh"

class GatePhaseSpaceBrentActor;
class GatePhaseSpaceBrentActorMessenger : public GatePhaseSpaceActorMessenger
{
public:
  GatePhaseSpaceBrentActorMessenger(GatePhaseSpaceBrentActor* sensor);
  virtual ~GatePhaseSpaceBrentActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GatePhaseSpaceBrentActor * pBrentActor;

  G4UIcmdWithABool* pEnablePrimaryEnergy;
  G4UIcmdWithAString* pCoordinateFrameCmd;

};

#endif /* end #define GatePhaseSpaceBrentACTORMESSENGER_HH*/
