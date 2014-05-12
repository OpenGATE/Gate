/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateElTubMessenger_h
#define GateElTubMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateElTub;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateElTubMessenger: public GateVolumeMessenger
{
  public:
    GateElTubMessenger(GateElTub* itsCreator);
   ~GateElTubMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    
    virtual inline GateElTub* GetElTubCreator() 
      { return (GateElTub*)GetVolumeCreator(); }

  private:
    G4UIcmdWithADoubleAndUnit* pElTubHeightCmd;
    G4UIcmdWithADoubleAndUnit* pElTubRshortCmd;
    G4UIcmdWithADoubleAndUnit* pElTubRlongCmd;
};

#endif

