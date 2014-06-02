/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateARFDataToRootMessenger_h
#define GateARFDataToRootMessenger_h 1

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateOutputModuleMessenger.hh"
#include "GateARFDataToRoot.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateARFDataToRootMessenger: public GateOutputModuleMessenger
{
  public:
    GateARFDataToRootMessenger(GateARFDataToRoot* GateARFDataToRoot);
   ~GateARFDataToRootMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

  protected:
    GateARFDataToRoot* m_GateARFDataToRoot;
    
    G4UIcmdWithAString* setARFDataFilecmd;
    G4UIcmdWithADoubleAndUnit* setDepth;
    G4UIcmdWithAString* smoothDRFcmd;
};

#endif
#endif

