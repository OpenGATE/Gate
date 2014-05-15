/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBlurringMessenger_h
#define GateBlurringMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include "GateVBlurringLaw.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateBlurring;

class GateBlurringMessenger: public GatePulseProcessorMessenger
{
  public:
    GateBlurringMessenger(GateBlurring* itsResolution);
    virtual ~GateBlurringMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateBlurring* GetBlurring()
      { return (GateBlurring*) GetPulseProcessor(); }

  private:
    GateVBlurringLaw* CreateBlurringLaw(const G4String& law);

    G4UIcmdWithAString *lawCmd;

};

#endif
