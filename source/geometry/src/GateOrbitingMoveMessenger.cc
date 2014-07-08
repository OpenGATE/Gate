/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateOrbitingMoveMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithoutParameter.hh"

//----------------------------------------------------------------------------------------
GateOrbitingMoveMessenger::GateOrbitingMoveMessenger(GateOrbitingMove* itsOrbitingMove)
  :GateObjectMoveMessenger(itsOrbitingMove)
{ 
    
    G4String cmdName;

    cmdName = GetDirectoryName()+"setPoint1";
    
    Point1Cmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    Point1Cmd->SetGuidance("Set the first point defining the orbit axis.");
    Point1Cmd->SetParameterName("X1","Y1","Z1",false);

    cmdName = GetDirectoryName()+"setPoint2";
    Point2Cmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    Point2Cmd->SetGuidance("Set the second point defining the orbit axis.");
    Point2Cmd->SetParameterName("X2","Y2","Z2",false);

    cmdName = GetDirectoryName()+"setSpeed"; 
    VelocityCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    VelocityCmd->SetGuidance("Set the orbiting angular speed.");
    VelocityCmd->SetParameterName("domega/dt",false);
    VelocityCmd->SetUnitCategory("Angular speed");

    cmdName = GetDirectoryName()+"enableAutoRotation";
    EnableAutoRotationCmd = new G4UIcmdWithABool(cmdName,this);
    EnableAutoRotationCmd->SetGuidance("Enable the object auto-rotation option.");
    EnableAutoRotationCmd->SetParameterName("flag",true);
    EnableAutoRotationCmd->SetDefaultValue(true);

    cmdName = GetDirectoryName()+"disableAutoRotation";
    DisableAutoRotationCmd = new G4UIcmdWithABool(cmdName,this);
    DisableAutoRotationCmd->SetGuidance("Enable the object auto-rotation option.");
    DisableAutoRotationCmd->SetParameterName("flag",true);
    DisableAutoRotationCmd->SetDefaultValue(true);
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GateOrbitingMoveMessenger::~GateOrbitingMoveMessenger()
{
    delete EnableAutoRotationCmd;
    delete DisableAutoRotationCmd;
    delete Point1Cmd;
    delete Point2Cmd;
    delete VelocityCmd;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GateOrbitingMoveMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{ 
  
  if( command==Point1Cmd )
    { 
      GetOrbitingMove()->SetPoint1(Point1Cmd->GetNew3VectorValue(newValue));/*TellGeometryToUpdate();*/}   
  
  else if( command==Point2Cmd )
    { 
      GetOrbitingMove()->SetPoint2(Point2Cmd->GetNew3VectorValue(newValue)); /*TellGeometryToUpdate();*/}   
  
  else if( command==VelocityCmd )
    { 
      GetOrbitingMove()->SetVelocity(VelocityCmd->GetNewDoubleValue(newValue));  /*TellGeometryToUpdate();*/}   
  
  else if ( command==EnableAutoRotationCmd )
    { GetOrbitingMove()->SetAutoRotation(EnableAutoRotationCmd->GetNewBoolValue(newValue)); /*TellGeometryToUpdate();*/}   

  else if ( command==DisableAutoRotationCmd )
    { GetOrbitingMove()->SetAutoRotation( !(DisableAutoRotationCmd->GetNewBoolValue(newValue)) ); /*TellGeometryToUpdate();*/}   

  else 
    GateObjectMoveMessenger::SetNewValue(command,newValue);
  
}
//----------------------------------------------------------------------------------------
