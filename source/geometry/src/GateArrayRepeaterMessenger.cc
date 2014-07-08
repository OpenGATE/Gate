/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateArrayRepeaterMessenger.hh"
#include "GateArrayRepeater.hh"


#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateArrayRepeaterMessenger::GateArrayRepeaterMessenger(GateArrayRepeater* itsCubicArrayRepeater)
  :GateObjectRepeaterMessenger(itsCubicArrayRepeater)
{ 
//    G4cout << " ***** Passage dans le constructeur de GateArrayRepeaterMessenger" << G4endl;
    
    G4String cmdName;

    cmdName = GetDirectoryName()+"setRepeatVector";
    SetRepeatVectorCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    SetRepeatVectorCmd->SetGuidance("Set the repetition vector (from the center of one copy to the center of the next one).");
    SetRepeatVectorCmd->SetParameterName("dX","dY","dZ",false);
    SetRepeatVectorCmd->SetUnitCategory("Length");

    cmdName = GetDirectoryName()+"setRepeatNumberX";
    SetRepeatNumberXCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetRepeatNumberXCmd->SetGuidance("Set the number of copies of the object along X.");
    SetRepeatNumberXCmd->SetParameterName("Nx",false);
    SetRepeatNumberXCmd->SetRange("Nx >= 1");

    cmdName = GetDirectoryName()+"setRepeatNumberY";
    SetRepeatNumberYCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetRepeatNumberYCmd->SetGuidance("Set the number of copies of the object along Y.");
    SetRepeatNumberYCmd->SetParameterName("Ny",false);
    SetRepeatNumberYCmd->SetRange("Ny >= 1");

    cmdName = GetDirectoryName()+"setRepeatNumberZ";
    SetRepeatNumberZCmd = new G4UIcmdWithAnInteger(cmdName,this);
    SetRepeatNumberZCmd->SetGuidance("Set the number of copies of the object along Z.");
    SetRepeatNumberZCmd->SetParameterName("Nz",false);
    SetRepeatNumberZCmd->SetRange("Nz >= 1");

    cmdName = GetDirectoryName()+"autoCenter";
    AutoCenterCmd = new G4UIcmdWithABool(cmdName,this);
    AutoCenterCmd->SetGuidance("Enable or disable auto-centering.");
    AutoCenterCmd->SetParameterName("flag",true);
    AutoCenterCmd->SetDefaultValue(true);

 }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateArrayRepeaterMessenger::~GateArrayRepeaterMessenger()
{
    delete AutoCenterCmd;
    delete SetRepeatVectorCmd;
    delete SetRepeatNumberXCmd;
    delete SetRepeatNumberYCmd;
    delete SetRepeatNumberZCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateArrayRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
//  G4cout << " Cubic array new value" << G4endl;
  
  if( command==SetRepeatVectorCmd )
    { 
//     G4cout << " repeat number = " << SetRepeatVectorCmd->GetNew3VectorValue(newValue) << G4endl;
      
     GetCubicArrayRepeater()->SetRepeatVector(SetRepeatVectorCmd->GetNew3VectorValue(newValue));/*TellGeometryToUpdate();*/}   
  else if( command==SetRepeatNumberXCmd )
    { GetCubicArrayRepeater()->SetRepeatNumberX(SetRepeatNumberXCmd->GetNewIntValue(newValue)); /*TellGeometryToRebuild();*/}   
  else if( command==SetRepeatNumberYCmd )
    { GetCubicArrayRepeater()->SetRepeatNumberY(SetRepeatNumberYCmd->GetNewIntValue(newValue)); /*TellGeometryToRebuild();*/}   
  else if( command==SetRepeatNumberZCmd )
    { GetCubicArrayRepeater()->SetRepeatNumberZ(SetRepeatNumberZCmd->GetNewIntValue(newValue)); /*TellGeometryToRebuild();*/}   
  else if( command==AutoCenterCmd )
    { GetCubicArrayRepeater()->SetAutoCenterFlag(AutoCenterCmd->GetNewBoolValue(newValue)); /*TellGeometryToUpdate();*/}   
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
