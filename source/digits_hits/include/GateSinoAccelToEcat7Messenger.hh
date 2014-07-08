/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_ECAT7

#ifndef GateSinoAccelToEcat7Messenger_h
#define GateSinoAccelToEcat7Messenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateSinoAccelToEcat7;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;




class GateSinoAccelToEcat7Messenger: public GateOutputModuleMessenger
{
  public:
    GateSinoAccelToEcat7Messenger(GateSinoAccelToEcat7* gateSinoAccelToEcat7);
   ~GateSinoAccelToEcat7Messenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateSinoAccelToEcat7*       m_gateSinoAccelToEcat7;

    G4UIcmdWithAString*         SetFileNameCmd;
    G4UIcmdWithAnInteger*       SetMashingCmd;
    G4UIcmdWithAnInteger*       SetSpanCmd;
    G4UIcmdWithAnInteger*       SetMaxRingDiffCmd;
    G4UIcmdWithAnInteger*       SetEcatAccelCameraNumberCmd;
};

#endif
#endif
