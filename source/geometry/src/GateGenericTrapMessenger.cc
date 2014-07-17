/*
 * GateGenericTrapMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateGenericTrapMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateGenericTrap.hh"

GateGenericTrapMessenger::GateGenericTrapMessenger(GateGenericTrap* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setVertex1";
  GenericTrapVertex1Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex1Cmd->SetGuidance("Set vertex #1 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex1Cmd->SetParameterName("Vertex1x", "Vertex1y", "Vertex1z", false);
  GenericTrapVertex1Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex2";
  GenericTrapVertex2Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex2Cmd->SetGuidance("Set vertex #2 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex2Cmd->SetParameterName("Vertex2x", "Vertex2y", "Vertex2z", false);
  GenericTrapVertex2Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex3";
  GenericTrapVertex3Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex3Cmd->SetGuidance("Set vertex #3 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex3Cmd->SetParameterName("Vertex3x", "Vertex3y", "Vertex3z", false);
  GenericTrapVertex3Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex4";
  GenericTrapVertex4Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex4Cmd->SetGuidance("Set vertex #4 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex4Cmd->SetParameterName("Vertex4x", "Vertex4y", "Vertex4z", false);
  GenericTrapVertex4Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex5";
  GenericTrapVertex5Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex5Cmd->SetGuidance("Set vertex #5 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex5Cmd->SetParameterName("Vertex5x", "Vertex5y", "Vertex5z", false);
  GenericTrapVertex5Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex6";
  GenericTrapVertex6Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex6Cmd->SetGuidance("Set vertex #6 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex6Cmd->SetParameterName("Vertex6x", "Vertex6y", "Vertex6z", false);
  GenericTrapVertex6Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex7";
  GenericTrapVertex7Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex7Cmd->SetGuidance("Set vertex #7 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex7Cmd->SetParameterName("Vertex7x", "Vertex7y", "Vertex7z", false);
  GenericTrapVertex7Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex8";
  GenericTrapVertex8Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  GenericTrapVertex8Cmd->SetGuidance("Set vertex #8 of the generic trapezoid (z is omitted!)");
  GenericTrapVertex8Cmd->SetParameterName("Vertex8x", "Vertex8y", "Vertex8z", false);
  GenericTrapVertex8Cmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  GenericTrapZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  GenericTrapZLengthCmd->SetGuidance("Set Z height of the generic trapezoid");
  GenericTrapZLengthCmd->SetParameterName("ZLength", false);
  GenericTrapZLengthCmd->SetRange("ZLength>0.");
  GenericTrapZLengthCmd->SetUnitCategory("Length");
}

GateGenericTrapMessenger::~GateGenericTrapMessenger()
{
  delete GenericTrapVertex1Cmd;
  delete GenericTrapVertex2Cmd;
  delete GenericTrapVertex3Cmd;
  delete GenericTrapVertex4Cmd;
  delete GenericTrapVertex5Cmd;
  delete GenericTrapVertex6Cmd;
  delete GenericTrapVertex7Cmd;
  delete GenericTrapVertex8Cmd;
  delete GenericTrapZLengthCmd;
}

void GateGenericTrapMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == GenericTrapVertex1Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(0, GenericTrapVertex1Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex2Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(1, GenericTrapVertex2Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex3Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(2, GenericTrapVertex3Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex4Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(3, GenericTrapVertex4Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex5Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(4, GenericTrapVertex5Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex6Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(5, GenericTrapVertex6Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex7Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(6, GenericTrapVertex7Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapVertex8Cmd ) {
      GetGenericTrapCreator()->SetGenericTrapVertex(7, GenericTrapVertex8Cmd->GetNew3VectorValue(newValue));
  } else if (command == GenericTrapZLengthCmd) {
      GetGenericTrapCreator()->SetGenericTrapZLength(GenericTrapZLengthCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
