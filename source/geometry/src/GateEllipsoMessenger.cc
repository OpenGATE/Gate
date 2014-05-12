/*-----------------------------------
-----------------------------------*/




#include "GateEllipsoMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateEllipso.hh" 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateEllipsoMessenger::GateEllipsoMessenger(GateEllipso *itsCreator)
  :GateVolumeMessenger(itsCreator)
{
  
  G4String dir = GetDirectoryName()+"geometry/";

  G4String cmdName;

  cmdName = dir+"setXLength";
  EllipsoHalfxCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipsoHalfxCmd->SetGuidance("Set half x length of the ellipsoid");
  EllipsoHalfxCmd->SetParameterName("Halfx", false);
  EllipsoHalfxCmd->SetRange("Halfx>=0.");
  EllipsoHalfxCmd->SetUnitCategory("Length");

  cmdName = dir+"setYLength";
  EllipsoHalfyCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipsoHalfyCmd->SetGuidance("Set half y length of the ellipsoid");
  EllipsoHalfyCmd->SetParameterName("Halfy", false);
  EllipsoHalfyCmd->SetRange("Halfy>=0.");
  EllipsoHalfyCmd->SetUnitCategory("Length");

  cmdName = dir+"setZLength";
  EllipsoHalfzCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipsoHalfzCmd->SetGuidance("Set half z length of the ellipsoid");
  EllipsoHalfzCmd->SetParameterName("Halfz", false);
  EllipsoHalfzCmd->SetRange("Halfz>=0.");
  EllipsoHalfzCmd->SetUnitCategory("Length");

  cmdName = dir+"setZBottomCut";
  EllipsoBottomCutzCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipsoBottomCutzCmd->SetGuidance("Set position of the bottom cut, 0 if no cut");
  EllipsoBottomCutzCmd->SetParameterName("BottomCutz", false);
  EllipsoBottomCutzCmd->SetUnitCategory("Length");

  cmdName = dir+"setZTopCut";
  EllipsoTopCutzCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipsoTopCutzCmd->SetGuidance("Set position of the bottom cut, 0 if no cut");
  EllipsoTopCutzCmd->SetParameterName("TopCutz", false);
  EllipsoTopCutzCmd->SetUnitCategory("Length");

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateEllipsoMessenger::~GateEllipsoMessenger()
{
  delete EllipsoHalfxCmd;
  delete EllipsoHalfyCmd;
  delete EllipsoHalfzCmd;
  delete EllipsoBottomCutzCmd;
  delete EllipsoTopCutzCmd;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateEllipsoMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if(command==EllipsoHalfxCmd)
    {GetEllipsoCreator()->SetEllipsopxSemiAxis(EllipsoHalfxCmd->GetNewDoubleValue(newValue));}
  else if(command==EllipsoHalfyCmd)
    {GetEllipsoCreator()->SetEllipsopySemiAxis(EllipsoHalfyCmd->GetNewDoubleValue(newValue));}
  else if(command==EllipsoHalfzCmd)
    {GetEllipsoCreator()->SetEllipsopzSemiAxis(EllipsoHalfzCmd->GetNewDoubleValue(newValue));}
  else if(command==EllipsoBottomCutzCmd)
    {GetEllipsoCreator()->SetEllipsopzBottomCut(EllipsoBottomCutzCmd->GetNewDoubleValue(newValue));}
  else if(command==EllipsoTopCutzCmd)
    {GetEllipsoCreator()->SetEllipsopzTopCut(EllipsoTopCutzCmd->GetNewDoubleValue(newValue));}
  else
    GateVolumeMessenger::SetNewValue(command, newValue);

}
