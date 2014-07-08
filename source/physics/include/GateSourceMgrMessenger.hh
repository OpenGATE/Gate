/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceMgrMessenger_h
#define GateSourceMgrMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateSourceMgr.hh"

class GateClock;
class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;

#include "GateUIcmdWithAVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateSourceMgrMessenger: public G4UImessenger
{
public:
  GateSourceMgrMessenger(GateSourceMgr*);
  ~GateSourceMgrMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
    
private:
  GateSourceMgr*                     m_sourceMgr;
    
  G4UIdirectory*                       GateSourceDir;
  G4UIcmdWithAString*                  SelectSourceCmd;
  GateUIcmdWithAVector<G4String>*      AddSourceCmd;
  G4UIcmdWithAString*                  RemoveSourceCmd;
  G4UIcmdWithoutParameter*             ListSourcesCmd;
  G4UIcmdWithAnInteger*                VerboseCmd;
  //G4UIcmdWithAnInteger*                UseAutoWeightCmd;
};

#endif

