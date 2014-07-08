/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTrpdMessenger_h
#define GateTrpdMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateTrpd;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateTrpdMessenger: public GateVolumeMessenger
{
public:
  GateTrpdMessenger(GateTrpd* itsCreator);
  ~GateTrpdMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
  // get here the pointer to created object from base class VObjectCreator see doxygen
  virtual inline GateTrpd* GetTrpdCreator() 
  { return (GateTrpd*) GetVolumeCreator(); }
  
private:
  G4UIcmdWithADoubleAndUnit* TrpdX1LengthCmd;    
  G4UIcmdWithADoubleAndUnit* TrpdY1LengthCmd;    
  G4UIcmdWithADoubleAndUnit* TrpdX2LengthCmd;    
  G4UIcmdWithADoubleAndUnit* TrpdY2LengthCmd;    
  G4UIcmdWithADoubleAndUnit* TrpdZLengthCmd;     
  //  G4UIcmdWith3VectorAndUnit* TrpdBoxLengthCmd
  //  G4UIcmdWith3VectorAndUnit* TrpdBoxPosCmd
  G4UIcmdWithADoubleAndUnit* TrpdXBoxLengthCmd;  
  G4UIcmdWithADoubleAndUnit* TrpdYBoxLengthCmd;  
  G4UIcmdWithADoubleAndUnit* TrpdZBoxLengthCmd;  
  G4UIcmdWithADoubleAndUnit* TrpdXBoxPosCmd;     
  G4UIcmdWithADoubleAndUnit* TrpdYBoxPosCmd;     
  G4UIcmdWithADoubleAndUnit* TrpdZBoxPosCmd;     
};

#endif

