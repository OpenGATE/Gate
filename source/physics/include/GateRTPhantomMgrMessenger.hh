#ifndef GateRTPhantomMgrMessenger_h
#define GateRTPhantomMgrMessenger_h 1

#include "GateMessenger.hh"

#include "GateRTPhantomMgr.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateRTPhantomMgrMessenger: public GateMessenger
{
public:
  GateRTPhantomMgrMessenger(GateRTPhantomMgr* PhMgr);
  ~GateRTPhantomMgrMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
    
private:
  GateRTPhantomMgr*                       m_PhMgr;
    
  G4UIcmdWithoutParameter*             DescribeCmd;
  G4UIcmdWithAnInteger*                VerboseCmd;
  G4UIcmdWithAString*                  insertCmd;

};

#endif

