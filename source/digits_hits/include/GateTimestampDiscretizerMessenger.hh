/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTimestampDiscretizerMessenger_h
#define GateTimestampDiscretizerMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;

class GateTimestampDiscretizer;

class GateTimestampDiscretizerMessenger: public GatePulseProcessorMessenger
{
  public:
    GateTimestampDiscretizerMessenger(GateTimestampDiscretizer* itsTimestampDiscretizer);
    virtual ~GateTimestampDiscretizerMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateTimestampDiscretizer* GetTimestampDiscretizer()
      { return (GateTimestampDiscretizer*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADoubleAndUnit   *frequencyCmd;
    G4UIcmdWithADoubleAndUnit   *timeCmd;
};

#endif
