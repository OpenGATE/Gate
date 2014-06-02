/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateUpholderMessenger_h
#define GateUpholderMessenger_h 1

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

class GateUpholder;

class GateUpholderMessenger: public GatePulseProcessorMessenger
{
  public:
    GateUpholderMessenger(GateUpholder* itsUpholder);
    virtual ~GateUpholderMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateUpholder* GetUpholder()
      { return (GateUpholder*) GetPulseProcessor(); }

  private:
    G4UIcmdWithADoubleAndUnit   *upholdCmd;
};

#endif
