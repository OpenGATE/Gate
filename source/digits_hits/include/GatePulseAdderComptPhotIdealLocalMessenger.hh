

#ifndef GatePulseAdderComptPhotIdealLocalMessenger_h
#define GatePulseAdderComptPhotIdealLocalMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GatePulseAdderComptPhotIdealLocal;


class GatePulseAdderComptPhotIdealLocalMessenger: public GatePulseProcessorMessenger
{
  public:
    GatePulseAdderComptPhotIdealLocalMessenger(GatePulseAdderComptPhotIdealLocal* itsPulseAdder);
     ~GatePulseAdderComptPhotIdealLocalMessenger();

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GatePulseAdderComptPhotIdealLocal* GetPulseAdderComptPhotIdealLocal()
      { return (GatePulseAdderComptPhotIdealLocal*) GetPulseProcessor(); }

    private:
      G4UIcmdWithAString   *newVolCmd;
      std::vector<G4String> m_name;

};

#endif
