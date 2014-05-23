/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTrapMessenger_h
#define GateTrapMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateTrap;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateTrapMessenger: public GateVolumeMessenger
{
  public:
    GateTrapMessenger(GateTrap* itsCreator);
   ~GateTrapMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateTrap* GetTrapCreator()
      { return (GateTrap*) GetVolumeCreator(); }

  private:
    G4UIcmdWithADoubleAndUnit* TrapDzCmd;
    G4UIcmdWithADoubleAndUnit* TrapThetaCmd;
    G4UIcmdWithADoubleAndUnit* TrapPhiCmd;
    G4UIcmdWithADoubleAndUnit* TrapDy1Cmd;
    G4UIcmdWithADoubleAndUnit* TrapDx1Cmd;
    G4UIcmdWithADoubleAndUnit* TrapDx2Cmd;
    G4UIcmdWithADoubleAndUnit* TrapAlp1Cmd;
    G4UIcmdWithADoubleAndUnit* TrapDy2Cmd;
    G4UIcmdWithADoubleAndUnit* TrapDx3Cmd;
    G4UIcmdWithADoubleAndUnit* TrapDx4Cmd;
    G4UIcmdWithADoubleAndUnit* TrapAlp2Cmd;
};

#endif

