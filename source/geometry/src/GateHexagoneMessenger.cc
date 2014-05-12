/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateHexagoneMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateHexagone.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateHexagoneMessenger::GateHexagoneMessenger(GateHexagone *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 

  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName = dir + "setRadius";
  HexagoneRadiusCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  HexagoneRadiusCmd->SetGuidance("Set radius of hexagone.");
  HexagoneRadiusCmd->SetParameterName("Radius",false);
  HexagoneRadiusCmd->SetRange("Radius>0.");
  HexagoneRadiusCmd->SetUnitCategory("Length");

  
  cmdName = dir + "setHeight";
  G4cout << " GetDirectoryName()+ setHeight = "  << cmdName << G4endl;
  HexagoneHeightCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  HexagoneHeightCmd->SetGuidance("Set height of the hexagone.");
  HexagoneHeightCmd->SetParameterName("Height",false);
  HexagoneHeightCmd->SetRange("Height>0.");
  HexagoneHeightCmd->SetUnitCategory("Length");

  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateHexagoneMessenger::~GateHexagoneMessenger()
{
    delete HexagoneHeightCmd;
    delete HexagoneRadiusCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateHexagoneMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command == HexagoneRadiusCmd )
    { GetHexagoneCreator()->SetHexagoneRadius(HexagoneRadiusCmd->GetNewDoubleValue(newValue));}
  else if( command==HexagoneHeightCmd )
    { GetHexagoneCreator()->SetHexagoneHeight(HexagoneHeightCmd->GetNewDoubleValue(newValue));}   
  else
    GateVolumeMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
