/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateRotationMoveMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithoutParameter.hh"

GateRotationMoveMessenger::GateRotationMoveMessenger(GateRotationMove* itsRotationMove)
  :GateObjectRepeaterMessenger(itsRotationMove)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setAxis";
    RotationAxisCmd = new G4UIcmdWith3Vector(cmdName,this);
    RotationAxisCmd->SetGuidance("Set the rotation axis.");
    RotationAxisCmd->SetParameterName("cosAlpha","cosBeta","cosGamma",false);
    RotationAxisCmd->SetRange("cosAlpha != 0 || cosBeta != 0 || cosGamma != 0");

    cmdName = GetDirectoryName()+"setSpeed";
    VelocityCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    VelocityCmd->SetGuidance("Set the rotation angular speed.");
    VelocityCmd->SetParameterName("domega/dt",false);
    VelocityCmd->SetUnitCategory("Angular speed");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateRotationMoveMessenger::~GateRotationMoveMessenger()
{
    delete RotationAxisCmd;
    delete VelocityCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateRotationMoveMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==RotationAxisCmd )
    { GetRotationMove()->SetRotationAxis(RotationAxisCmd->GetNew3VectorValue(newValue));}   
  
  else if( command==VelocityCmd )
    { GetRotationMove()->SetVelocity(VelocityCmd->GetNewDoubleValue(newValue));}   
  
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
