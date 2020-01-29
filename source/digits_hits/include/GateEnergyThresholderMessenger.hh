/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateEnergyThresholderMessenger_h
#define GateEnergyThresholderMessenger_h 1

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

class GateEnergyThresholder;

class GateEnergyThresholderMessenger: public GatePulseProcessorMessenger
{
  public:
    GateEnergyThresholderMessenger(GateEnergyThresholder* itsEnergyThresholder);
    virtual ~GateEnergyThresholderMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateEnergyThresholder* GetEnergyThresholder()
      { return (GateEnergyThresholder*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADoubleAndUnit   *thresholdCmd;
     GateVEffectiveEnergyLaw* CreateEffectiveEnergyLaw(const G4String& law);
     G4UIcmdWithAString *lawCmd;
};

#endif
