/*
 * GateParaMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateParaMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GatePara.hh"

GateParaMessenger::GateParaMessenger(GatePara* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setXLength";
  ParaXLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaXLengthCmd->SetGuidance("Set x length of the parallelepiped");
  ParaXLengthCmd->SetParameterName("XLength", false);
  ParaXLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setYLength";
  ParaYLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaYLengthCmd->SetGuidance("Set y length of the parallelepiped");
  ParaYLengthCmd->SetParameterName("YLength", false);
  ParaYLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  ParaZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaZLengthCmd->SetGuidance("Set z length of the parallelepiped");
  ParaZLengthCmd->SetParameterName("ZLength", false);
  ParaZLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setAlpha";
  ParaAlphaCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaAlphaCmd->SetGuidance("Set alpha angle of the parallelepiped");
  ParaAlphaCmd->SetParameterName("Alpha", true);
  ParaAlphaCmd->SetDefaultValue(0*degree);
  ParaAlphaCmd->SetUnitCategory("Angle");

  cmdName = dir + "setTheta";
  ParaThetaCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaThetaCmd->SetGuidance("Set polar angle of the parallelepiped");
  ParaThetaCmd->SetParameterName("Theta", true);
  ParaThetaCmd->SetDefaultValue(0*degree);
  ParaThetaCmd->SetUnitCategory("Angle");

  cmdName = dir + "setPhi";
  ParaPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaPhiCmd->SetGuidance("Set azimuthal angle of the parallelepiped");
  ParaPhiCmd->SetParameterName("Phi", true);
  ParaPhiCmd->SetDefaultValue(0*degree);
  ParaPhiCmd->SetUnitCategory("Angle");
}

GateParaMessenger::~GateParaMessenger()
{
  delete ParaXLengthCmd;
  delete ParaYLengthCmd;
  delete ParaZLengthCmd;
  delete ParaAlphaCmd;
  delete ParaThetaCmd;
  delete ParaPhiCmd;
}

void GateParaMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == ParaXLengthCmd ) {
      GetParaCreator()->SetParaXLength(ParaXLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaYLengthCmd) {
      GetParaCreator()->SetParaYLength(ParaYLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaZLengthCmd) {
      GetParaCreator()->SetParaZLength(ParaZLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaAlphaCmd) {
      GetParaCreator()->SetParaAlpha(ParaAlphaCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaThetaCmd) {
      GetParaCreator()->SetParaTheta(ParaThetaCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaPhiCmd) {
      GetParaCreator()->SetParaPhi(ParaPhiCmd->GetNewDoubleValue(newValue));

  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
