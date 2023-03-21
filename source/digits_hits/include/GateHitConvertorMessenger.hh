/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
//GND ClassToRemove


#ifndef GateHitConvertorMessenger_h
#define GateHitConvertorMessenger_h 1

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

class GateHitConvertor;

class GateHitConvertorMessenger: public GateClockDependentMessenger
{
  public:
    GateHitConvertorMessenger(GateHitConvertor* itsHitConvertor);
    inline ~GateHitConvertorMessenger() {}

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateHitConvertor* GetHitConvertor()
      { return (GateHitConvertor*) GetClockDependent(); }

};

#endif
