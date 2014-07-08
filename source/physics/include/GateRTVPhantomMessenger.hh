#ifndef GateRTVPhantomMessenger_h
#define GateRTVPhantomMessenger_h 1

#include "GateMessenger.hh"

class GateRTVPhantom;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithoutParameter;
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateRTVPhantomMessenger: public GateMessenger
{
public:
  GateRTVPhantomMessenger(GateRTVPhantom* Ph);
  ~GateRTVPhantomMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
    
private:
  GateRTVPhantom*                       m_RTVPhantom;

    G4UIcmdWithAnInteger*       SetRTVPhantomTPFCmd;

    G4UIcmdWithAString*              SetRTVPhantomCmd;

    G4UIcmdWithAString*              SetRTVPhantomHeaderFileCmd;

    G4UIcmdWithADoubleAndUnit*        SetTPFCmd;

    G4UIcmdWithoutParameter* SetAttAsActCmd;
    G4UIcmdWithoutParameter* SetActAsAttCmd;
};

#endif


