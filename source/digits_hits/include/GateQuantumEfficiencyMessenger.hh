/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateQuantumEfficiencyMessenger_h
#define GateQuantumEfficiencyMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include <vector>
#include "G4UIdirectory.hh"

class G4UIcmdWithAString;
class G4UIcmdWithADouble;

class GateQuantumEfficiency;

class GateQuantumEfficiencyMessenger: public GatePulseProcessorMessenger
{
  public:
    GateQuantumEfficiencyMessenger(GateQuantumEfficiency* itsQE);
    virtual ~GateQuantumEfficiencyMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateQuantumEfficiency* GetQE()
      { return (GateQuantumEfficiency*) GetPulseProcessor(); }

  private:
    G4UIcmdWithAString   *newVolCmd;
    G4UIcmdWithADouble   *uniqueQECmd;
    G4UIcmdWithAString   *newFileCmd;
};

#endif
