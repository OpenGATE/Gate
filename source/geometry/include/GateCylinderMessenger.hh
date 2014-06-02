/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCylinderMessenger_h
#define GateCylinderMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateCylinder;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateCylinderMessenger: public GateVolumeMessenger
{
  public:
    GateCylinderMessenger(GateCylinder* itsCreator);
   ~GateCylinderMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateCylinder* GetCylinderCreator() 
      { return (GateCylinder*)GetVolumeCreator(); }

  private:
    G4UIcmdWithADoubleAndUnit* pCylinderHeightCmd;
    G4UIcmdWithADoubleAndUnit* pCylinderRminCmd;
    G4UIcmdWithADoubleAndUnit* pCylinderRmaxCmd;
    G4UIcmdWithADoubleAndUnit* pCylinderSPhiCmd;
    G4UIcmdWithADoubleAndUnit* pCylinderDPhiCmd;
};

#endif

