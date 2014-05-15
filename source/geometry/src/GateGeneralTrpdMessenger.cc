/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateGeneralTrpdMessenger.hh"

#include "G4UIdirectory.hh"
// #include "G4UIcmdWithAString.hh"
// #include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
// #include "G4UIcmdWith3VectorAndUnit.hh"
// #include "G4UIcmdWithoutParameter.hh"

#include "GateGeneralTrpd.hh"

//-------------------------------------------------------------------------------------
GateGeneralTrpdMessenger::GateGeneralTrpdMessenger(GateGeneralTrpd *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir+"setX1Length";
  TrpdX1LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdX1LengthCmd->SetGuidance("Set half length along X of the plane at -dz position");
  TrpdX1LengthCmd->SetParameterName("Length",false);
  TrpdX1LengthCmd->SetRange("Length>0.");
  TrpdX1LengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setX2Length";
  TrpdX2LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdX2LengthCmd->SetGuidance("Set half length along X of the plane at -dz position");
  TrpdX2LengthCmd->SetParameterName("Length",false);
  TrpdX2LengthCmd->SetRange("Length>0.");
  TrpdX2LengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setX3Length";
  TrpdX3LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdX3LengthCmd->SetGuidance("Set half length along X of the plane at -dz position");
  TrpdX3LengthCmd->SetParameterName("Length",false);
  TrpdX3LengthCmd->SetRange("Length>0.");
  TrpdX3LengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setX4Length";
  TrpdX4LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdX4LengthCmd->SetGuidance("Set half length along X of the plane at -dz position");
  TrpdX4LengthCmd->SetParameterName("Length",false);
  TrpdX4LengthCmd->SetRange("Length>0.");
  TrpdX4LengthCmd->SetUnitCategory("Length");

  cmdName = dir+"setY1Length";
  TrpdY1LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdY1LengthCmd->SetGuidance("Set half length along Y of the plane at -dz position");
  TrpdY1LengthCmd->SetParameterName("Length",false);
  TrpdY1LengthCmd->SetRange("Length>0.");
  TrpdY1LengthCmd->SetUnitCategory("Length");

   cmdName = dir+"setY2Length";
  TrpdY2LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdY2LengthCmd->SetGuidance("Set half length along X of the plane at +dz position");
  TrpdY2LengthCmd->SetParameterName("Length",false);
  TrpdY2LengthCmd->SetRange("Length>0.");
  TrpdY2LengthCmd->SetUnitCategory("Length");

    cmdName = dir+"setZLength";
  TrpdZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdZLengthCmd->SetGuidance("Set half length along Z of the trapezoid ");
  TrpdZLengthCmd->SetParameterName("Length",false);
  TrpdZLengthCmd->SetRange("Length>0.");
  TrpdZLengthCmd->SetUnitCategory("Length");

    cmdName = dir+"setTheta";
  TrpdThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdThetaCmd->SetGuidance("Set half length along Theta of the trapeThetaoid ");
  TrpdThetaCmd->SetParameterName("Angle",false);
//  TrpdThetahCmd->SetRange("Length>0.");
  TrpdThetaCmd->SetUnitCategory("Angle");

    cmdName = dir+"setPhi";
  TrpdPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdPhiCmd->SetGuidance("Set half length along Phi of the trapezoid ");
  TrpdPhiCmd->SetParameterName("Angle",false);
//  TrpdPhihCmd->SetRange("Length>0.");
  TrpdPhiCmd->SetUnitCategory("Angle");

    cmdName = dir+"setAlp1";
  TrpdAlp1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdAlp1Cmd->SetGuidance("Set half length along Alp1 of the trapezoid ");
  TrpdAlp1Cmd->SetParameterName("Angle",false);
//  TrpdAlp1Cmd->SetRange("Length>0.");
  TrpdAlp1Cmd->SetUnitCategory("Angle");


    cmdName = dir+"setAlp2";
  TrpdAlp2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdAlp2Cmd->SetGuidance("Set half length along Alp2 of the trapezoid ");
  TrpdAlp2Cmd->SetParameterName("Angle",false);
//  TrpdAlp2Cmd->SetRange("Length>0.");
  TrpdAlp2Cmd->SetUnitCategory("Angle");

/*  cmdName = GetDirectoryName()+"setXBoxLength";
  TrpdXBoxLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdXBoxLengthCmd->SetGuidance("Set half length along X of the extruded box.");
  TrpdXBoxLengthCmd->SetParameterName("Length",false);
  TrpdXBoxLengthCmd->SetRange("Length>0.");
  TrpdXBoxLengthCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setYBoxLength";
  TrpdYBoxLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdYBoxLengthCmd->SetGuidance("Set half length along Y of the extruded box.");
  TrpdYBoxLengthCmd->SetParameterName("Length",false);
  TrpdYBoxLengthCmd->SetRange("Length>0.");
  TrpdYBoxLengthCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZBoxLength";
  TrpdZBoxLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdZBoxLengthCmd->SetGuidance("Set half length along Z of the extruded box.");
  TrpdZBoxLengthCmd->SetParameterName("Length",false);
  TrpdZBoxLengthCmd->SetRange("Length>0.");
  TrpdZBoxLengthCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setXBoxPos";
  TrpdXBoxPosCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdXBoxPosCmd->SetGuidance("Set center position X of the box.");
  TrpdXBoxPosCmd->SetParameterName("Length",false);
  TrpdXBoxPosCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setYBoxPos";
  TrpdYBoxPosCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdYBoxPosCmd->SetGuidance("Set center position Y of the box.");
  TrpdYBoxPosCmd->SetParameterName("Length",false);
  TrpdYBoxPosCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZBoxPos";
  TrpdZBoxPosCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrpdZBoxPosCmd->SetGuidance("Set center position Z of the box.");
  TrpdZBoxPosCmd->SetParameterName("Length",false);
  TrpdZBoxPosCmd->SetUnitCategory("Length");
*/
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
GateGeneralTrpdMessenger::~GateGeneralTrpdMessenger()
{
  delete  TrpdX1LengthCmd;    
  delete  TrpdX2LengthCmd;    
  delete  TrpdX3LengthCmd;    
  delete  TrpdX4LengthCmd;    
  delete  TrpdY1LengthCmd;    
  delete  TrpdY2LengthCmd;    
  delete  TrpdZLengthCmd;     
/*  delete  TrpdXBoxLengthCmd;  
  delete  TrpdYBoxLengthCmd;  
  delete  TrpdZBoxLengthCmd;  
  delete  TrpdXBoxPosCmd;     
  delete  TrpdYBoxPosCmd;     
  delete  TrpdZBoxPosCmd;     */
  delete  TrpdThetaCmd;
  delete  TrpdAlp2Cmd;
  delete  TrpdAlp1Cmd;
  delete  TrpdPhiCmd;


}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
void GateGeneralTrpdMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if(      command==TrpdX1LengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdX1Length(TrpdX1LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  else if( command==TrpdX2LengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdX2Length(TrpdX2LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  else if( command==TrpdX3LengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdX3Length(TrpdX3LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  else if( command==TrpdX4LengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdX4Length(TrpdX4LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   

  else if( command==TrpdY1LengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdY1Length(TrpdY1LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
 
  else if( command==TrpdY2LengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdY2Length(TrpdY2LengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdZLengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdZLength(TrpdZLengthCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdThetaCmd )
    { GetGeneralTrpdCreator()->SetTrpdTheta(TrpdThetaCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdPhiCmd )
    { GetGeneralTrpdCreator()->SetTrpdPhi(TrpdPhiCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdAlp1Cmd )
    { GetGeneralTrpdCreator()->SetTrpdAlp1(TrpdAlp1Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
  else if( command==TrpdAlp2Cmd )
    { GetGeneralTrpdCreator()->SetTrpdAlp2(TrpdAlp2Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}   
  
/*
Loic  else if( command==TrpdXBoxLengthCmd )
    { GetGeneralTrpdCreator()->SetTrpdTrudXLength(TrpdXBoxLengthCmd->GetNewDoubleValue(newValue)); TellGeometryToRebuild();}   
  			                   
  else if( command==TrpdYBoxLengthCmd )	                   
    { GetGeneralTrpdCreator()->SetTrpdTrudYLength(TrpdYBoxLengthCmd->GetNewDoubleValue(newValue)); TellGeometryToRebuild();}   
  			                   
  else if( command==TrpdZBoxLengthCmd )	                   
    { GetGeneralGeneralTrpdCreator()->SetTrpdTrudZLength(TrpdZBoxLengthCmd->GetNewDoubleValue(newValue)); TellGeometryToRebuild();}   
  
  else if( command==TrpdXBoxPosCmd )
    { GetGeneralTrpdCreator()->SetTrpdTrudXPos(TrpdXBoxPosCmd->GetNewDoubleValue(newValue)); TellGeometryToRebuild();}   
  			                   
  else if( command==TrpdYBoxPosCmd )	                   
    { GetGeneralTrpdCreator()->SetTrpdTrudYPos(TrpdYBoxPosCmd->GetNewDoubleValue(newValue)); TellGeometryToRebuild();}   
  			                   
  else if( command==TrpdZBoxPosCmd )	                   
    { GetGeneralTrpdCreator()->SetTrpdTrudZPos(TrpdZBoxPosCmd->GetNewDoubleValue(newValue)); TellGeometryToRebuild();}   
*/

  else
    GateVolumeMessenger::SetNewValue(command,newValue);

}
//-------------------------------------------------------------------------------------
