/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*----------------------
   Modifications history

     Gate 6.2

	C. Comtat, CEA/SHFJ, 10/02/2011	   Allows for an interfile-like ("ecat8") output instead of ecat7.
					   It does not require the ecat library! (GATE_USE_ECAT7 not set)
----------------------*/

#include "GateConfiguration.h"

#ifndef GateSinoToEcat7Messenger_h
#define GateSinoToEcat7Messenger_h 1

#include "GateOutputModuleMessenger.hh"

class GateSinoToEcat7;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;




class GateSinoToEcat7Messenger: public GateOutputModuleMessenger
{
  public:
    GateSinoToEcat7Messenger(GateSinoToEcat7* gateSinoToEcat7);
   ~GateSinoToEcat7Messenger();

    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateSinoToEcat7*             m_gateSinoToEcat7;

    G4UIcmdWithAString*          SetFileNameCmd;
    G4UIcmdWithAnInteger*        SetMashingCmd;
    G4UIcmdWithAnInteger*        SetSpanCmd;
    G4UIcmdWithAnInteger*        SetMaxRingDiffCmd;
    G4UIcmdWithAnInteger*        SetEcatCameraNumberCmd;
    G4UIcmdWithAString*          SetIsotopeCodeCmd;
    G4UIcmdWithADoubleAndUnit*   SetIsotopeHalflifeCmd;
    G4UIcmdWithADouble*          SetIsotopeBranchingFractionCmd;
    #ifdef GATE_USE_ECAT7
    G4UIcmdWithAnInteger*        SetEcatVersionCmd;
    #endif
};
#endif
