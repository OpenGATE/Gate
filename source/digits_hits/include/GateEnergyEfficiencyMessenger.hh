/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateEnergyEfficiencyMessenger_h
#define GateEnergyEfficiencyMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;

class GateEnergyEfficiency;

class GateEnergyEfficiencyMessenger: public GatePulseProcessorMessenger
{
  public:
    GateEnergyEfficiencyMessenger(GateEnergyEfficiency* itsPulseProcessor);
    virtual ~GateEnergyEfficiencyMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithAString   *distNameCmd;
};

#endif
