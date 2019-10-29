

#ifndef GateCC3DlocalSpblurringMessenger_h
#define GateCC3DlocalSpblurringMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "G4UIdirectory.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateCC3DlocalSpblurring;

class GateCC3DlocalSpblurringMessenger: public GatePulseProcessorMessenger
{
public:
    GateCC3DlocalSpblurringMessenger(GateCC3DlocalSpblurring* itsResolution);
    virtual ~GateCC3DlocalSpblurringMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GateCC3DlocalSpblurring* GetLocal3DSpBlurring()
    { return (GateCC3DlocalSpblurring*) GetPulseProcessor(); }

private:
    G4UIcmdWithAString   *newVolCmd;
    std::vector<G4UIdirectory*> m_volDirectory;
    std::vector<G4UIcmdWith3VectorAndUnit*>   sigmaCmd;
    std::vector<G4String> m_name;
    G4int m_count;
};

#endif
