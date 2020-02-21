/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateDoIModelsMessenger_h
#define GateDoIModelsMessenger_h 1

#include "GatePulseProcessorMessenger.hh"
#include "GateVDoILaw.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateDoIModels;

class GateDoIModelsMessenger: public GatePulseProcessorMessenger
{
public:
    GateDoIModelsMessenger(GateDoIModels* itsDoIModel);
    virtual ~GateDoIModelsMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateDoIModels* GetDoIModel()
    { return (GateDoIModels*) GetPulseProcessor(); }

private:
    G4UIcmdWith3Vector *axisCmd;
    GateVDoILaw* CreateDoILaw(const G4String& law);
    G4UIcmdWithAString *lawCmd;
};

#endif
