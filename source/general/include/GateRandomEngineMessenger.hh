/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateRandomEngineMessenger_h
#define GateRandomEngineMessenger_h 1

#include "GateMessenger.hh"
#include "GateRandomEngine.hh"
#include "CLHEP/Random/RandomEngine.h"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class GateRandomEngine;

class GateRandomEngineMessenger: public GateMessenger
{

public:
  GateRandomEngineMessenger(GateRandomEngine* gateRandomEngine);
  ~GateRandomEngineMessenger();
  void SetNewValue(G4UIcommand*, G4String);

private:
  G4UIcmdWithAString* GetEngineNameCmd;
  G4UIcmdWithAString* GetEngineSeedCmd;
  G4UIcmdWithAString* GetEngineFromFileCmd; //TC
  G4UIcmdWithAnInteger* GetEngineVerboseCmd;
  G4UIcmdWithoutParameter* ShowEngineStatus;
  GateRandomEngine* m_gateRandomEngine;
};

#endif
