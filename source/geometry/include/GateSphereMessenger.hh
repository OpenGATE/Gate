/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSphereMessenger_h
#define GateSphereMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateSphere;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateSphereMessenger: public GateVolumeMessenger
{
  public:
    GateSphereMessenger(GateSphere* itsCreator);
   ~GateSphereMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateSphere* GetSphereCreator() 
      { return (GateSphere*)GetVolumeCreator(); }

  private:
    G4UIcmdWithADoubleAndUnit* SphereRminCmd;
    G4UIcmdWithADoubleAndUnit* SphereRmaxCmd;
    G4UIcmdWithADoubleAndUnit* SphereSPhiCmd;
    G4UIcmdWithADoubleAndUnit* SphereDPhiCmd;
    G4UIcmdWithADoubleAndUnit* SphereSThetaCmd;
    G4UIcmdWithADoubleAndUnit* SphereDThetaCmd;
};

#endif

