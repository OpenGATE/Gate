/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateTrapMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateTrap.hh"

class GateVolumeMessenger;


//----------------------------------------------------------------------------------------------------------
GateTrapMessenger::GateTrapMessenger(GateTrap *itsCreator)
  :GateVolumeMessenger(itsCreator)
{ 
  G4String cmdName;
  
  cmdName = GetDirectoryName()+"geometry/setDz";
  TrapDzCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDzCmd->SetGuidance("Set Dz of Trap.");
  TrapDzCmd->SetParameterName("Dz",false);
  TrapDzCmd->SetRange("Dz>0.");
  TrapDzCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setTheta";
  TrapThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapThetaCmd->SetGuidance("Set Theta of Trap.");
  TrapThetaCmd->SetParameterName("Theta",false);
  TrapThetaCmd->SetUnitCategory("Angle");

  cmdName = GetDirectoryName()+"geometry/setPhi";
  TrapPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapPhiCmd->SetGuidance("Set Phi of Trap.");
  TrapPhiCmd->SetParameterName("Phi",false);
  TrapPhiCmd->SetUnitCategory("Angle");

  cmdName = GetDirectoryName()+"geometry/setDy1";
  TrapDy1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDy1Cmd->SetGuidance("Set Dy1 of Trap.");
  TrapDy1Cmd->SetParameterName("Dy1",false);
  TrapDy1Cmd->SetRange("Dy1>0.");
  TrapDy1Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setDx1";
  TrapDx1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDx1Cmd->SetGuidance("Set Dx1 of Trap.");
  TrapDx1Cmd->SetParameterName("Dx1",false);
  TrapDx1Cmd->SetRange("Dx1>0.");
  TrapDx1Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setDx2";
  TrapDx2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDx2Cmd->SetGuidance("Set Dx2 of Trap.");
  TrapDx2Cmd->SetParameterName("Dx2",false);
  TrapDx2Cmd->SetRange("Dx2>0.");
  TrapDx2Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setAlp1";
  TrapAlp1Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapAlp1Cmd->SetGuidance("Set Alp1 of Trap.");
  TrapAlp1Cmd->SetParameterName("Alp1",false);
  TrapAlp1Cmd->SetUnitCategory("Angle");

  cmdName = GetDirectoryName()+"geometry/setDy2";
  TrapDy2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDy2Cmd->SetGuidance("Set Dy2 of Trap.");
  TrapDy2Cmd->SetParameterName("Dy2",false);
  TrapDy2Cmd->SetRange("Dy2>0.");
  TrapDy2Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setDx3";
  TrapDx3Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDx3Cmd->SetGuidance("Set Dx3 of Trap.");
  TrapDx3Cmd->SetParameterName("Dx3",false);
  TrapDx3Cmd->SetRange("Dx3>0.");
  TrapDx3Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setDx4";
  TrapDx4Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapDx4Cmd->SetGuidance("Set Dx4 of Trap.");
  TrapDx4Cmd->SetParameterName("Dx4",false);
  TrapDx4Cmd->SetRange("Dx4>0.");
  TrapDx4Cmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"geometry/setAlp2";
  TrapAlp2Cmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(),this);
  TrapAlp2Cmd->SetGuidance("Set Alp2 of Trap.");
  TrapAlp2Cmd->SetParameterName("Alp2",false);
  TrapAlp2Cmd->SetUnitCategory("Angle");
}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
GateTrapMessenger::~GateTrapMessenger()
{
    delete TrapDzCmd;
    delete TrapThetaCmd;
    delete TrapPhiCmd;
    delete TrapDy1Cmd;
    delete TrapDx1Cmd;
    delete TrapDx2Cmd;
    delete TrapAlp1Cmd;
    delete TrapDy2Cmd;
    delete TrapDx3Cmd;
    delete TrapDx4Cmd;
    delete TrapAlp2Cmd;
}
//----------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------
void GateTrapMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command == TrapDzCmd )
    { GetTrapCreator()->SetTrapDz(TrapDzCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapThetaCmd )
    { GetTrapCreator()->SetTrapTheta(TrapThetaCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapPhiCmd )
    { GetTrapCreator()->SetTrapPhi(TrapPhiCmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapDy1Cmd )
    { GetTrapCreator()->SetTrapDy1(TrapDy1Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapDx1Cmd )
    { GetTrapCreator()->SetTrapDx1(TrapDx1Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapDx2Cmd )
    { GetTrapCreator()->SetTrapDx2(TrapDx2Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapAlp1Cmd )
    { GetTrapCreator()->SetTrapAlp1(TrapAlp1Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapDy2Cmd )
    { GetTrapCreator()->SetTrapDy2(TrapDy2Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapDx3Cmd )
    { GetTrapCreator()->SetTrapDx3(TrapDx3Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapDx4Cmd )
    { GetTrapCreator()->SetTrapDx4(TrapDx4Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else if( command==TrapAlp2Cmd )
    { GetTrapCreator()->SetTrapAlp2(TrapAlp2Cmd->GetNewDoubleValue(newValue)); /*TellGeometryToRebuild();*/}
  else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
//----------------------------------------------------------------------------------------------------------
