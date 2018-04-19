

#ifndef GateStripSpatialDiscretizationMessenger_h
#define GateStripSpatialDiscretizationMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateStripSpatialDiscretization;


class GateStripSpatialDiscretizationMessenger: public GatePulseProcessorMessenger
{
  public:
    GateStripSpatialDiscretizationMessenger(GateStripSpatialDiscretization* itsPulseAdder);
    virtual ~GateStripSpatialDiscretizationMessenger();

     inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline  void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GateStripSpatialDiscretization* GetStripSpatialDiscretization()
      { return (GateStripSpatialDiscretization*) GetPulseProcessor(); }

 private:

    G4UIcmdWithAString   *newVolCmd;
     std::vector<G4UIdirectory*> m_volDirectory;

    std::vector<G4UIcmdWithADoubleAndUnit*>    pthresholdCmd;
    std::vector<G4UIcmdWithADoubleAndUnit*>    pStripOffsetX;
    std::vector<G4UIcmdWithADoubleAndUnit*>    pStripOffsetY;
    std::vector<G4UIcmdWithAnInteger*>    pNumberStripsX;
    std::vector<G4UIcmdWithAnInteger*>     pNumberStripsY;

    std::vector<G4String> m_name;
     G4int m_count;

};

#endif
