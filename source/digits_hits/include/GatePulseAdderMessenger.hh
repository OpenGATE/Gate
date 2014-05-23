/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseAdderMessenger_h
#define GatePulseAdderMessenger_h 1

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

class GatePulseAdder;

/*! \class  GatePulseAdderMessenger
    \brief  Messenger for the GatePulseAdder

    - GatePulseAdderMessenger - by Daniel.Strul@iphe.unil.ch

    \sa GatePulseAdder, GatePulseProcessorMessenger
*/
class GatePulseAdderMessenger: public GatePulseProcessorMessenger
{
  public:
    GatePulseAdderMessenger(GatePulseAdder* itsPulseAdder);
    inline ~GatePulseAdderMessenger() {}

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GatePulseAdder* GetPulseAdder()
      { return (GatePulseAdder*) GetPulseProcessor(); }

};

#endif
