/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateThresholderMessenger_h
#define GateThresholderMessenger_h 1

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

class GateThresholder;

class GateThresholderMessenger: public GatePulseProcessorMessenger
{
  public:
    GateThresholderMessenger(GateThresholder* itsThresholder);
    virtual ~GateThresholderMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateThresholder* GetThresholder()
      { return (GateThresholder*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADoubleAndUnit   *thresholdCmd;
};

#endif
