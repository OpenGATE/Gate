/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseAdderGpuSpectMessenger_h
#define GatePulseAdderGpuSpectMessenger_h 1

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

class GatePulseAdderGPUSpect;


class GatePulseAdderGPUSpectMessenger: public GatePulseProcessorMessenger
{
  public:
    GatePulseAdderGPUSpectMessenger(GatePulseAdderGPUSpect* itsPulseAdder);
    inline ~GatePulseAdderGPUSpectMessenger() {}

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GatePulseAdderGPUSpect* GetPulseAdderGPUSpect()
      { return (GatePulseAdderGPUSpect*) GetPulseProcessor(); }

};

#endif
