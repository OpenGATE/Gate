/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateNoiseMessenger_h
#define GateNoiseMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;

class GateNoise;

class GateNoiseMessenger: public GatePulseProcessorMessenger
{
  public:
    GateNoiseMessenger(GateNoise* itsPulseProcessor);
    virtual ~GateNoiseMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithAString        *m_deltaTDistribCmd;
    G4UIcmdWithAString        *m_energyDistribCmd;
};

#endif
