/*-----------------------------------
-----------------------------------*/


#ifndef GateEllipsoMessenger_h
#define GateEllipsoMessenger_h 1

#include "globals.hh"

#include "GateVolumeMessenger.hh"

class GateEllipso;


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

class GateEllipsoMessenger: public GateVolumeMessenger
{
public:
  GateEllipsoMessenger(GateEllipso* itsCreator);
  ~GateEllipsoMessenger();

  void SetNewValue(G4UIcommand*, G4String);

  virtual inline GateEllipso* GetEllipsoCreator()
  { return (GateEllipso*)GetVolumeCreator();}

private:
  G4UIcmdWithADoubleAndUnit* EllipsoHalfxCmd;
  G4UIcmdWithADoubleAndUnit* EllipsoHalfyCmd;
  G4UIcmdWithADoubleAndUnit* EllipsoHalfzCmd;
  G4UIcmdWithADoubleAndUnit* EllipsoBottomCutzCmd;
  G4UIcmdWithADoubleAndUnit* EllipsoTopCutzCmd;

};

#endif


