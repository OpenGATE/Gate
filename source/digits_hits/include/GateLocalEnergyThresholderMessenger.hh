/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateLocalEnergyThresholderMessenger_h
#define GateLocalEnergyThresholderMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include "GateVEffectiveEnergyLaw.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateLocalEnergyThresholder;

class GateLocalEnergyThresholderMessenger: public GatePulseProcessorMessenger
{
  public:
    GateLocalEnergyThresholderMessenger(GateLocalEnergyThresholder* itsEnergyThresholder);
    virtual ~GateLocalEnergyThresholderMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);
      inline void SetNewValue2(G4UIcommand* aCommand, G4String aString);

    inline GateLocalEnergyThresholder* GetLocalEnergyThresholder()
      { return (GateLocalEnergyThresholder*) GetPulseProcessor(); }

private:

    GateVEffectiveEnergyLaw* CreateEffectiveEnergyLaw(const G4String& law, int i);
    G4UIcmdWithAString   *newVolCmd;

    std::vector<G4UIdirectory*> m_volDirectory;
    std::vector<G4UIcmdWithAString*>  lawCmd;
    std::vector<G4UIcmdWithADoubleAndUnit*> thresholdCmd;
    std::vector<G4String> m_name;
    G4int m_count;
};

#endif
