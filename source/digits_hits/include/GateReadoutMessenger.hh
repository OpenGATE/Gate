/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateReadoutMessenger_h
#define GateReadoutMessenger_h 1

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

class GateReadout;

/*! \class  GateReadoutMessenger
    \brief  Messenger for the GateReadout

    - GateReadoutMessenger - by Daniel.Strul@iphe.unil.ch

    \sa GateReadout, GatePulseProcessorMessenger
*/
class GateReadoutMessenger: public GatePulseProcessorMessenger
{
  public:
    GateReadoutMessenger(GateReadout* itsReadout);
    ~GateReadoutMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateReadout* GetReadout()
      { return (GateReadout*) GetPulseProcessor(); }

  private:
    G4UIcmdWithAnInteger*      SetDepthCmd;
};

#endif
