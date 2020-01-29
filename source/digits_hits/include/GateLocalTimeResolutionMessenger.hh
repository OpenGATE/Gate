

#ifndef GateLocalTimeResolutionMessenger_h
#define GateLocalTimeResolutionMessenger_h 1

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

class GateLocalTimeResolution;

class GateLocalTimeResolutionMessenger: public GatePulseProcessorMessenger
{
public:
  GateLocalTimeResolutionMessenger(GateLocalTimeResolution* itsResolution);
  virtual ~GateLocalTimeResolutionMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
  inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

  inline GateLocalTimeResolution* GetLocalTimeResolution()
  { return (GateLocalTimeResolution*) GetPulseProcessor(); }

private:
  G4UIcmdWithAString   *newVolCmd;
  std::vector<G4UIdirectory*> m_volDirectory;
  std::vector<G4UIcmdWithADoubleAndUnit*>   timeResolutionCmd;
  std::vector<G4String> m_name;
  G4int m_count;
};

#endif
