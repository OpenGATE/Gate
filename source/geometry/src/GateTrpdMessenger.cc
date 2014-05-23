/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTrpdMessenger.hh"

#include "G4UIdirectory.hh"
// #include "G4UIcmdWithAString.hh"
// #include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
// #include "G4UIcmdWith3VectorAndUnit.hh"
// #include "G4UIcmdWithoutParameter.hh"

#include "GateTrpd.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateTrpdMessenger::GateTrpdMessenger(GateTrpd *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 

  G4String dir = GetDirectoryName() + "geometry/";
  
  G4String cmdName = dir+"setX1Length";
  TrpdX1LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdX1LengthCmd->SetGuidance("Set half length along X of the plane at -dz position");
  TrpdX1LengthCmd->SetParameterName("Length",false);
  TrpdX1LengthCmd->SetRange("Length>0.");
  TrpdX1LengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setY1Length";
  TrpdY1LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdY1LengthCmd->SetGuidance("Set half length along Y of the plane at -dz position");
  TrpdY1LengthCmd->SetParameterName("Length",false);
  TrpdY1LengthCmd->SetRange("Length>0.");
  TrpdY1LengthCmd->SetUnitCategory("Length");

   cmdName = dir+"setX2Length";
  TrpdX2LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdX2LengthCmd->SetGuidance("Set half length along X of the plane at +dz position");
  TrpdX2LengthCmd->SetParameterName("Length",false);
  TrpdX2LengthCmd->SetRange("Length>0.");
  TrpdX2LengthCmd->SetUnitCategory("Length");

   cmdName = dir+"setY2Length";
  TrpdY2LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdY2LengthCmd->SetGuidance("Set half length along Y of the plane at +dz position");
  TrpdY2LengthCmd->SetParameterName("Length",false);
  TrpdY2LengthCmd->SetRange("Length>0.");
  TrpdY2LengthCmd->SetUnitCategory("Length");

   cmdName = dir+"setZLength";
  TrpdZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdZLengthCmd->SetGuidance("Set half length along Z of the trapezoid ");
  TrpdZLengthCmd->SetParameterName("Length",false);
  TrpdZLengthCmd->SetRange("Length>0.");
  TrpdZLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setXBoxLength";
  TrpdXBoxLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdXBoxLengthCmd->SetGuidance("Set half length along X of the extruded box.");
  TrpdXBoxLengthCmd->SetParameterName("Length",false);
  TrpdXBoxLengthCmd->SetRange("Length>0.");
  TrpdXBoxLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setYBoxLength";
  TrpdYBoxLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdYBoxLengthCmd->SetGuidance("Set half length along Y of the extruded box.");
  TrpdYBoxLengthCmd->SetParameterName("Length",false);
  TrpdYBoxLengthCmd->SetRange("Length>0.");
  TrpdYBoxLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setZBoxLength";
  TrpdZBoxLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdZBoxLengthCmd->SetGuidance("Set half length along Z of the extruded box.");
  TrpdZBoxLengthCmd->SetParameterName("Length",false);
  TrpdZBoxLengthCmd->SetRange("Length>0.");
  TrpdZBoxLengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setXBoxPos";
  TrpdXBoxPosCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdXBoxPosCmd->SetGuidance("Set center position X of the box.");
  TrpdXBoxPosCmd->SetParameterName("Length",false);
  TrpdXBoxPosCmd->SetUnitCategory("Length");

  cmdName = dir+"setYBoxPos";
  TrpdYBoxPosCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdYBoxPosCmd->SetGuidance("Set center position Y of the box.");
  TrpdYBoxPosCmd->SetParameterName("Length",false);
  TrpdYBoxPosCmd->SetUnitCategory("Length");

  cmdName = dir+"setZBoxPos";
  TrpdZBoxPosCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdZBoxPosCmd->SetGuidance("Set center position Z of the box.");
  TrpdZBoxPosCmd->SetParameterName("Length",false);
  TrpdZBoxPosCmd->SetUnitCategory("Length");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateTrpdMessenger::~GateTrpdMessenger()
{
  delete  TrpdX1LengthCmd;    
  delete  TrpdY1LengthCmd;    
  delete  TrpdX2LengthCmd;    
  delete  TrpdY2LengthCmd;    
  delete  TrpdZLengthCmd;     
  delete  TrpdXBoxLengthCmd;  
  delete  TrpdYBoxLengthCmd;  
  delete  TrpdZBoxLengthCmd;  
  delete  TrpdXBoxPosCmd;     
  delete  TrpdYBoxPosCmd;     
  delete  TrpdZBoxPosCmd;     
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateTrpdMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if(      command==TrpdX1LengthCmd )
    { GetTrpdCreator()->SetTrpdX1Length(TrpdX1LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdY1LengthCmd )
    { GetTrpdCreator()->SetTrpdY1Length(TrpdY1LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdX2LengthCmd )
    { GetTrpdCreator()->SetTrpdX2Length(TrpdX2LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdY2LengthCmd )
    { GetTrpdCreator()->SetTrpdY2Length(TrpdY2LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdZLengthCmd )
    { GetTrpdCreator()->SetTrpdZLength(TrpdZLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdXBoxLengthCmd )
    { GetTrpdCreator()->SetTrpdTrudXLength(TrpdXBoxLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  			                   
  else if( command==TrpdYBoxLengthCmd )	                   
    { GetTrpdCreator()->SetTrpdTrudYLength(TrpdYBoxLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  			                   
  else if( command==TrpdZBoxLengthCmd )	                   
    { GetTrpdCreator()->SetTrpdTrudZLength(TrpdZBoxLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdXBoxPosCmd )
    { GetTrpdCreator()->SetTrpdTrudXPos(TrpdXBoxPosCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  			                   
  else if( command==TrpdYBoxPosCmd )	                   
    { GetTrpdCreator()->SetTrpdTrudYPos(TrpdYBoxPosCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  			                   
  else if( command==TrpdZBoxPosCmd )	                   
    { GetTrpdCreator()->SetTrpdTrudZPos(TrpdZBoxPosCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else
    GateVolumeMessenger::SetNewValue(command,newValue);

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
