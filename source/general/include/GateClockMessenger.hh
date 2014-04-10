/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateClockMessenger_h
#define GateClockMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateClockMessenger: public G4UImessenger
{
public:
  GateClockMessenger();
  ~GateClockMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  G4UIdirectory*             pGateTimingDir;
  G4UIcmdWithADoubleAndUnit* pTimeCmd;
  G4UIcmdWithAnInteger*      pVerboseCmd;
};

#endif
