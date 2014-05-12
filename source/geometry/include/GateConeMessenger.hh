/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateConeMessenger_h
#define GateConeMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateCone;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateConeMessenger: public GateVolumeMessenger
{
  public:
    GateConeMessenger(GateCone* itsCreator);
   ~GateConeMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateCone* GetConeCreator() 
      { return (GateCone*)GetVolumeCreator(); }

  private:
    G4UIcmdWithADoubleAndUnit* ConeHeightCmd;
    G4UIcmdWithADoubleAndUnit* ConeRmin1Cmd;
    G4UIcmdWithADoubleAndUnit* ConeRmax1Cmd;
    G4UIcmdWithADoubleAndUnit* ConeRmin2Cmd;
    G4UIcmdWithADoubleAndUnit* ConeRmax2Cmd;
    G4UIcmdWithADoubleAndUnit* ConeSPhiCmd;
    G4UIcmdWithADoubleAndUnit* ConeDPhiCmd;
};

#endif

