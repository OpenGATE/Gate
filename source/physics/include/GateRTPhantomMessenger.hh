
#ifndef GateRTPhantomMessenger_h
#define GateRTPhantomMessenger_h

#include "GateMessenger.hh"

class GateRTPhantom;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateRTPhantomMessenger : public GateMessenger {
public:
  GateRTPhantomMessenger(GateRTPhantom *Ph);
  ~GateRTPhantomMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  GateRTPhantom*                       m_Ph;

  G4UIcmdWithoutParameter*             DescribeCmd;
  G4UIcmdWithAnInteger*                VerboseCmd;
  G4UIcmdWithAString*                  attachCmd;
  G4UIcmdWithAString*                  attachSCmd;
  G4UIcmdWithoutParameter*             DisableCmd;

};

#endif
