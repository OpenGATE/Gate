/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateEccentRotMoveMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithoutParameter.hh"

//------------------------------------------------------------------------------------------------------------
GateEccentRotMoveMessenger::GateEccentRotMoveMessenger(GateEccentRotMove* itsEccentRotMove)
  :GateObjectRepeaterMessenger(itsEccentRotMove)
{ 
    G4String cmdName;

    cmdName = GetDirectoryName()+"setShiftXYZ";
    ShiftCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
    ShiftCmd->SetGuidance("Set the shift on the X-Y-Z directions : params= X Y Z values with units");
    ShiftCmd->SetParameterName("X","Y","Z",false);

    cmdName = GetDirectoryName()+"setSpeed";
    VelocityCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    VelocityCmd->SetGuidance("Set the orbiting angular speed.");
    VelocityCmd->SetParameterName("domega/dt",false);
    VelocityCmd->SetUnitCategory("Angular speed");

}
//------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------
GateEccentRotMoveMessenger::~GateEccentRotMoveMessenger()
{
    delete ShiftCmd;
    delete VelocityCmd;
}
//------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------
void GateEccentRotMoveMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if( command==ShiftCmd )
    { GetEccentRotMove()->SetShift(ShiftCmd->GetNew3VectorValue(newValue));}   
  
  else if( command==VelocityCmd )
    { GetEccentRotMove()->SetVelocity(VelocityCmd->GetNewDoubleValue(newValue));}   
  
  else 
    GateObjectRepeaterMessenger::SetNewValue(command,newValue);
  
}
//------------------------------------------------------------------------------------------------------------
