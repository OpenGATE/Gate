/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseProcessorMessenger_h
#define GatePulseProcessorMessenger_h 1

#include "GateClockDependentMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateVPulseProcessor;



class GatePulseProcessorMessenger: public GateClockDependentMessenger
{
  public:
    GatePulseProcessorMessenger(GateVPulseProcessor* itsPulseProcessor);
   ~GatePulseProcessorMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

    inline GateVPulseProcessor* GetPulseProcessor() 
      { return (GateVPulseProcessor*) GetClockDependent(); }

  private:
};

#endif

