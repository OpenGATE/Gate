/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateHexagoneMessenger_h
#define GateHexagoneMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateHexagone;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateHexagoneMessenger: public GateVolumeMessenger
{
  public:
    GateHexagoneMessenger(GateHexagone* itsCreator);
   ~GateHexagoneMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateHexagone* GetHexagoneCreator() 
      { return (GateHexagone*)GetVolumeCreator(); }

  private:
    G4UIcmdWithADoubleAndUnit* HexagoneHeightCmd;
    G4UIcmdWithADoubleAndUnit* HexagoneRadiusCmd;
    
};

#endif

