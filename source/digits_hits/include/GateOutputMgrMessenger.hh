/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateOutputMgrMessenger_h
#define GateOutputMgrMessenger_h 1

#include "GateMessenger.hh"

class GateOutputMgr;

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

//class GateOutputMgrMessenger: public GateMessenger, GateActorMessenger
class GateOutputMgrMessenger: public GateMessenger
{
public:
  GateOutputMgrMessenger(GateOutputMgr* outputMgr);
  virtual ~GateOutputMgrMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  GateOutputMgr*                       m_outputMgr;
  G4UIdirectory*                       pGateOutputMess;
  G4UIcmdWithoutParameter*             DescribeCmd;
  G4UIcmdWithAnInteger*                VerboseCmd;
  G4UIcmdWithoutParameter*             AllowNoOutputCmd;
};

#endif
