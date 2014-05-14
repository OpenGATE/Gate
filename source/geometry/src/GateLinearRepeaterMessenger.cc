/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateLinearRepeaterMessenger.hh"
#include "GateLinearRepeater.hh"
#include "GateMessageManager.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//------------------------------------------------------------------------------------------------------------------
GateLinearRepeaterMessenger::GateLinearRepeaterMessenger(GateLinearRepeater* itsLinearRepeater)
  :GateObjectRepeaterMessenger(itsLinearRepeater)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setRepeatVector";
    SetRepeatVectorCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    SetRepeatVectorCmd->SetGuidance("Set the repetition vector (from the center of one copy to the center of the next one).");
    SetRepeatVectorCmd->SetParameterName("dX","dY","dZ",false);
    SetRepeatVectorCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName()+"setRepeatNumber";
    SetRepeatNumberCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetRepeatNumberCmd->SetGuidance("Set the number of copies of the object.");
    SetRepeatNumberCmd->SetParameterName("N",false);
    SetRepeatNumberCmd->SetRange("N >= 1");

    cmdName = GetDirectoryName()+"autoCenter";
    AutoCenterCmd = new G4UIcmdWithABool(cmdName,this);
    AutoCenterCmd->SetGuidance("Enable or disable auto-centering.");
    AutoCenterCmd->SetParameterName("flag",true);
    AutoCenterCmd->SetDefaultValue(true);

 }
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
GateLinearRepeaterMessenger::~GateLinearRepeaterMessenger()
{
    delete AutoCenterCmd;
    delete SetRepeatVectorCmd;
    delete SetRepeatNumberCmd;
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
void GateLinearRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  
  if( command==SetRepeatVectorCmd )

    {  GateMessage("Geometry", 5, "Linear repeater : repeat vector = " << SetRepeatVectorCmd->GetNew3VectorValue(newValue) << G4endl;);      
    GetLinearRepeater()->SetRepeatVector(SetRepeatVectorCmd->GetNew3VectorValue(newValue));}   
  else if( command==SetRepeatNumberCmd )
    { GateMessage("Geometry", 5, "Linear repeater : repeat number = " << SetRepeatNumberCmd->GetNewIntValue(newValue) << G4endl; ); 
    GetLinearRepeater()->SetRepeatNumber(SetRepeatNumberCmd->GetNewIntValue(newValue)); }   
  else if( command==AutoCenterCmd )
    { GetLinearRepeater()->SetAutoCenterFlag(AutoCenterCmd->GetNewBoolValue(newValue));}   
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}
//------------------------------------------------------------------------------------------------------------------
