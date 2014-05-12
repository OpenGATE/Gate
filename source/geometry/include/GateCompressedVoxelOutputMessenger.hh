/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCompressedVoxelOutputMessenger_h
#define GateCompressedVoxelOutputMessenger_h 1

#include "GateOutputModuleMessenger.hh"
#include "G4UIcmdWithABool.hh"

class GateCompressedVoxelOutput;
class G4UIcmdWithAString;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateCompressedVoxelOutputMessenger: public GateOutputModuleMessenger
{
  public:
    GateCompressedVoxelOutputMessenger(GateCompressedVoxelOutput* g);
   ~GateCompressedVoxelOutputMessenger();
    
    virtual void SetNewValue(G4UIcommand*, G4String);
    
  protected:
    GateCompressedVoxelOutput*          m_gateVoxelOutput;
    G4UIcmdWithAString*       SetFileNameCmd;
    G4UIcmdWithABool*         saveUncertaintyCmd;
   
};

#endif

