/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateARFTableMgrMessenger_h
#define GateARFTableMgrMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"

class GateARFTableMgr;
class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

#include "GateUIcmdWithAVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateARFTableMgrMessenger: public G4UImessenger
{
public:
  GateARFTableMgrMessenger(G4String aName, GateARFTableMgr*);
  ~GateARFTableMgrMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  GateARFTableMgr*             m_ARFTableMgr;

  G4UIdirectory*               GateARFTableDir;
  G4UIcmdWithAString*          cptTableEWCmd;
  G4UIcmdWithoutParameter*     ListARFTableCmd;
  G4UIcmdWithAnInteger*        VerboseCmd;
  G4UIcmdWithADouble*          setEResocmd;
  G4UIcmdWithADoubleAndUnit*   setERefcmd;
  G4UIcmdWithADoubleAndUnit*   setEThreshHoldcmd ;
  G4UIcmdWithADoubleAndUnit*   setEUpHoldcmd ;
  G4UIcmdWithAString*          SaveToBinaryFileCmd;
  G4UIcmdWithAnInteger*        SetNBinsCmd;
  G4UIcmdWithAString*          LoadFromBinaryFileCmd;
  G4UIcmdWithADoubleAndUnit*   setDistancecmd;
};

#endif
