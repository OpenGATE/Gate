/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVoxelOutputMessenger_h
#define GateVoxelOutputMessenger_h 1

#include "GateOutputModuleMessenger.hh"
#include "G4UIcmdWithABool.hh"

class GateVoxelOutput;
class G4UIcmdWithAString;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateVoxelOutputMessenger: public GateOutputModuleMessenger
{
  public:
    GateVoxelOutputMessenger(GateVoxelOutput* g);
   ~GateVoxelOutputMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
  protected:
    GateVoxelOutput*          m_gateVoxelOutput;
    G4UIcmdWithAString*       SetFileNameCmd;
    G4UIcmdWithABool*         saveUncertaintyCmd;
   
};

#endif

