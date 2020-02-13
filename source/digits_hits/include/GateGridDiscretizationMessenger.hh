

#ifndef GateGridDiscretizationMessenger_h
#define GateGridDiscretizationMessenger_h 1

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

class GateGridDiscretization;


class GateGridDiscretizationMessenger: public GatePulseProcessorMessenger
{
  public:
    GateGridDiscretizationMessenger(GateGridDiscretization* itsPulseAdder);
    virtual ~GateGridDiscretizationMessenger();

     inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
    inline  void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GateGridDiscretization* GetGridDiscretization()
      { return (GateGridDiscretization*) GetPulseProcessor(); }

 private:

    G4UIcmdWithAString   *newVolCmd;
     std::vector<G4UIdirectory*> m_volDirectory;

   // std::vector<G4UIcmdWithADoubleAndUnit*>    pthresholdCmd;
    std::vector<G4UIcmdWithADoubleAndUnit*>    pStripOffsetX;
    std::vector<G4UIcmdWithADoubleAndUnit*>    pStripOffsetY;
    std::vector<G4UIcmdWithADoubleAndUnit*>    pStripWidthX;
    std::vector<G4UIcmdWithADoubleAndUnit*>    pStripWidthY;
    std::vector<G4UIcmdWithAnInteger*>    pNumberStripsX;
    std::vector<G4UIcmdWithAnInteger*>     pNumberStripsY;
    std::vector<G4UIcmdWithAnInteger*>    pNumberReadOutBlocksX;
    std::vector<G4UIcmdWithAnInteger*>     pNumberReadOutBlocksY;
   // std::vector<G4UIcmdWithABool*>     pRejectionMultiplesCmd;

    std::vector<G4String> m_name;
     G4int m_count;

};

#endif
