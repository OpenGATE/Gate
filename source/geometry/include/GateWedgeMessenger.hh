/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateWedgeMessenger_h
#define GateWedgeMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateWedge;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateWedgeMessenger: public GateVolumeMessenger
{
  public:
    GateWedgeMessenger(GateWedge* itsCreator);
   ~GateWedgeMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateWedge* GetWedgeCreator() 
      { return (GateWedge*)GetVolumeCreator(); }

  private:
   G4UIcmdWithADoubleAndUnit* WedgeXLengthCmd;
   G4UIcmdWithADoubleAndUnit* WedgeNarrowerXLengthCmd;
   G4UIcmdWithADoubleAndUnit* WedgeYLengthCmd;
   G4UIcmdWithADoubleAndUnit* WedgeZLengthCmd;
};

#endif

