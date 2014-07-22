/*
 * GateTetraMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTetraMessenger.hh"

#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateTetra.hh"

GateTetraMessenger::GateTetraMessenger(GateTetra* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setVertex1";
  TetraP1Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  TetraP1Cmd->SetGuidance("Set vertex #1 of the tetrahedron");
  TetraP1Cmd->SetParameterName("Vertex1x","Vertex1y","Vertex1z", false);
  TetraP1Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex2";
  TetraP2Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  TetraP2Cmd->SetGuidance("Set vertex #2 of the tetrahedron");
  TetraP2Cmd->SetParameterName("Vertex2x","Vertex2y","Vertex2z", false);
  TetraP2Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex3";
  TetraP3Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  TetraP3Cmd->SetGuidance("Set vertex #3 of the tetrahedron");
  TetraP3Cmd->SetParameterName("Vertex3x","Vertex3y","Vertex3z", false);
  TetraP3Cmd->SetUnitCategory("Length");

  cmdName = dir + "setVertex4";
  TetraP4Cmd = new G4UIcmdWith3VectorAndUnit(cmdName.c_str(), this);
  TetraP4Cmd->SetGuidance("Set vertex #4 of the tetrahedron");
  TetraP4Cmd->SetParameterName("Vertex4x","Vertex4y","Vertex4z", false);
  TetraP4Cmd->SetUnitCategory("Length");
}

GateTetraMessenger::~GateTetraMessenger()
{
  delete TetraP1Cmd;
  delete TetraP2Cmd;
  delete TetraP3Cmd;
  delete TetraP4Cmd;
}

void GateTetraMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == TetraP1Cmd) {
      GetTetraCreator()->SetTetraP1(TetraP1Cmd->GetNew3VectorValue(newValue));
  } else if (command == TetraP2Cmd) {
      GetTetraCreator()->SetTetraP2(TetraP2Cmd->GetNew3VectorValue(newValue));
  } else if (command == TetraP3Cmd) {
      GetTetraCreator()->SetTetraP3(TetraP3Cmd->GetNew3VectorValue(newValue));
  } else if (command == TetraP4Cmd) {
      GetTetraCreator()->SetTetraP4(TetraP4Cmd->GetNew3VectorValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
