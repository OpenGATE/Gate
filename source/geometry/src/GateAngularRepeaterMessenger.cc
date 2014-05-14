/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateAngularRepeaterMessenger.hh"
#include "GateAngularRepeater.hh"


#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//---------------------------------------------------------------------------------------------------
GateAngularRepeaterMessenger::GateAngularRepeaterMessenger(GateAngularRepeater* itsAngularRepeater)
  :GateObjectRepeaterMessenger(itsAngularRepeater)
{ 
  G4String cmdName;

  cmdName = GetDirectoryName()+"setRepeatNumber";
  SetRepeatNumberCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetRepeatNumberCmd->SetGuidance("Set the number of copies of the object.");
  SetRepeatNumberCmd->SetParameterName("N",false);
  SetRepeatNumberCmd->SetRange("N >= 1");
  
  cmdName = GetDirectoryName()+"setPoint1";
  Point1Cmd = new G4UIcmdWith3Vector(cmdName,this);
  Point1Cmd->SetGuidance("Set the first point defining the repeater axis.");
  Point1Cmd->SetParameterName("X1","Y1","Z1",false);
  
  cmdName = GetDirectoryName()+"setPoint2";
  Point2Cmd = new G4UIcmdWith3Vector(cmdName,this);
  Point2Cmd->SetGuidance("Set the second point defining the repeater axis.");
  Point2Cmd->SetParameterName("X2","Y2","Z2",false);
  
  cmdName = GetDirectoryName()+"enableAutoRotation";
  EnableAutoRotationCmd = new G4UIcmdWithABool(cmdName,this);
  EnableAutoRotationCmd->SetGuidance("Enable the object auto-rotation option.");
  EnableAutoRotationCmd->SetParameterName("flag",true);
  EnableAutoRotationCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"disableAutoRotation";
  DisableAutoRotationCmd = new G4UIcmdWithABool(cmdName,this);
  DisableAutoRotationCmd->SetGuidance("Disable the object auto-rotation option.");
  DisableAutoRotationCmd->SetParameterName("flag",true);
  DisableAutoRotationCmd->SetDefaultValue(true);

  cmdName = GetDirectoryName()+"setFirstAngle";
  FirstAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  FirstAngleCmd->SetGuidance("Set the rotation angle for the first copy.");
  FirstAngleCmd->SetParameterName("phi0",false);
  FirstAngleCmd->SetUnitCategory("Angle");

  cmdName = GetDirectoryName()+"setAngularSpan";
  AngularSpanCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  AngularSpanCmd->SetGuidance("Set the rotation angle difference between the first and the last copy.");
  AngularSpanCmd->SetParameterName("Dphi",false);
  AngularSpanCmd->SetUnitCategory("Angle");

//    cmdName = GetDirectoryName()+"setAngularPitch";
//    AngularPitchCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
//    AngularPitchCmd->SetGuidance("Set the rotation angle difference between two adjacent copies.");
//    AngularPitchCmd->SetParameterName("dphi",false);
//    AngularPitchCmd->SetUnitCategory("Angle");

  cmdName = GetDirectoryName()+"setModuloNumber";// Modulo
  SetModuloNumberCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetModuloNumberCmd->SetGuidance("Set the number of objects in the periodic structure, and so the periodicity");
  SetModuloNumberCmd->SetParameterName("M",false);
  SetModuloNumberCmd->SetRange(" 1 <= M || M >=8");

  cmdName = GetDirectoryName()+"setZShift1";  // Shift1Cmd
  Shift1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift1Cmd->SetGuidance("Set the Z shift of modulo object 1");
  Shift1Cmd->SetParameterName("Z",false);
  Shift1Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift2";// Shift2Cmd;    
  Shift2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift2Cmd->SetGuidance("Set the Z shift of modulo object 2");
  Shift2Cmd->SetParameterName("Z",false);
  Shift2Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift3";    
  Shift3Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift3Cmd->SetGuidance("Set the Z shift of modulo object 3");
  Shift3Cmd->SetParameterName("Z",false);
  Shift3Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift4";
  Shift4Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift4Cmd->SetGuidance("Set the Z shift of modulo object 4");
  Shift4Cmd->SetParameterName("Z",false);
  Shift4Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift5";
  Shift5Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift5Cmd->SetGuidance("Set the Z shift of modulo object 5");
  Shift5Cmd->SetParameterName("Z",false);
  Shift5Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift6";
  Shift6Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift6Cmd->SetGuidance("Set the Z shift of modulo object 6");
  Shift6Cmd->SetParameterName("Z",false);
  Shift6Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift7";
  Shift7Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift7Cmd->SetGuidance("Set the Z shift of modulo object 7");
  Shift7Cmd->SetParameterName("Z",false);
  Shift7Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setZShift8";
  Shift8Cmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  Shift8Cmd->SetGuidance("Set the Z shift of modulo object 8");
  Shift8Cmd->SetParameterName("Z",false);
  Shift8Cmd->SetUnitCategory("Length");

}
//---------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------
GateAngularRepeaterMessenger::~GateAngularRepeaterMessenger()
{
    
    delete AngularSpanCmd;
    delete FirstAngleCmd;
    delete EnableAutoRotationCmd;
    delete DisableAutoRotationCmd;
    delete Point1Cmd;
    delete Point2Cmd;
    delete SetRepeatNumberCmd;
    delete SetModuloNumberCmd;       
    delete Shift1Cmd;
    delete Shift2Cmd;
    delete Shift3Cmd;
    delete Shift4Cmd;
    delete Shift5Cmd;
    delete Shift6Cmd;
    delete Shift7Cmd;
    delete Shift8Cmd;
    
}
//---------------------------------------------------------------------------------------------------


//---------------------------------------------------------------------------------------------------
void GateAngularRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
   if( command==Point1Cmd )
     { GetAngularRepeater()->SetPoint1(Point1Cmd->GetNew3VectorValue(newValue));}   
   else if( command==Point2Cmd )
     { GetAngularRepeater()->SetPoint2(Point2Cmd->GetNew3VectorValue(newValue));}   
   else if( command==SetRepeatNumberCmd )
     { GetAngularRepeater()->SetRepeatNumber(SetRepeatNumberCmd->GetNewIntValue(newValue));}   
   else if ( command==EnableAutoRotationCmd )
     { GetAngularRepeater()->SetAutoRotation(EnableAutoRotationCmd->GetNewBoolValue(newValue));}   
   else if ( command==DisableAutoRotationCmd )
     { GetAngularRepeater()->SetAutoRotation(!(DisableAutoRotationCmd->GetNewBoolValue(newValue)));} 
   else if ( command==FirstAngleCmd )
     { GetAngularRepeater()->SetFirstAngle( FirstAngleCmd->GetNewDoubleValue(newValue)) ;}   
   else if ( command==AngularSpanCmd )
     { GetAngularRepeater()->SetAngularSpan( AngularSpanCmd->GetNewDoubleValue(newValue)) ;}   
   else if ( command==SetModuloNumberCmd)
     { GetAngularRepeater()->SetModuloNumber( SetModuloNumberCmd->GetNewIntValue(newValue)) ;}   
   else if ( command==Shift1Cmd)
     { GetAngularRepeater()->SetZShift1(Shift1Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift2Cmd)
     { GetAngularRepeater()->SetZShift2(Shift2Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift3Cmd)
     { GetAngularRepeater()->SetZShift3(Shift3Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift4Cmd)
     { GetAngularRepeater()->SetZShift4(Shift4Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift5Cmd)
     { GetAngularRepeater()->SetZShift5(Shift5Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift6Cmd)
     { GetAngularRepeater()->SetZShift6(Shift6Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift7Cmd)
     { GetAngularRepeater()->SetZShift7(Shift7Cmd->GetNewDoubleValue(newValue)) ;}
   else if ( command==Shift8Cmd)
     { GetAngularRepeater()->SetZShift8(Shift8Cmd->GetNewDoubleValue(newValue)) ;}
   else
     {G4cout << "\n GateAngularRepeaterMesenger says: no cmd associated with :" << command <<"\n";}
}
//---------------------------------------------------------------------------------------------------
