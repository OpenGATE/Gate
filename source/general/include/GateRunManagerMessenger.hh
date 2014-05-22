/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateRunManagerMessenger_h
#define GateRunManagerMessenger_h 1

#include "G4UImessenger.hh"
class GateRunManager;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;

//-----------------------------------------------------------------------------
class GateRunManagerMessenger : public G4UImessenger
{
  public :
    GateRunManagerMessenger(GateRunManager* );
    virtual ~GateRunManagerMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
  private :
    GateRunManager* pRunManager;    
    G4UIcmdWithoutParameter* pRunInitCmd;
    G4UIcmdWithoutParameter* pRunGeomUpdateCmd;
    G4UIcmdWithABool* pRunEnableGlobalOutputCmd;  
};
//-----------------------------------------------------------------------------

#endif
