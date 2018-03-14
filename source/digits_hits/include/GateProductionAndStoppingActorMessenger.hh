/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
/*
  \class  GateProductionAndStoppingActorMessenger
  \author huisman@creatis.insa-lyon.fr
*/

#ifndef GateProductionAndStoppingActorMessenger_HH
#define GateProductionAndStoppingActorMessenger_HH

#include "globals.hh"
#include "GateImageActorMessenger.hh"

class G4UIcmdWithABool;
class G4UIcmdWithoutParameter;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;
class GateProductionAndStoppingActor;

class GateProductionAndStoppingActorMessenger : public GateImageActorMessenger
{
public:
  GateProductionAndStoppingActorMessenger(GateProductionAndStoppingActor* sensor);
  virtual ~GateProductionAndStoppingActorMessenger();

  void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  GateProductionAndStoppingActor * pActor;

  G4UIcmdWithAString* bCoordinateFrameCmd;

};

#endif /* end #define GateProductionAndStoppingActorMessenger_HH*/
#endif
