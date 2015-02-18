/*
 * GateTwistedBoxMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedBoxMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateTwistedBox.hh"

GateTwistedBoxMessenger::GateTwistedBoxMessenger(GateTwistedBox* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setXLength";
  TwistedBoxXLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedBoxXLengthCmd->SetGuidance("Set X extent of the twisted box");
  TwistedBoxXLengthCmd->SetParameterName("XLength", false);
  TwistedBoxXLengthCmd->SetRange("XLength>0.");
  TwistedBoxXLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setYLength";
  TwistedBoxYLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedBoxYLengthCmd->SetGuidance("Set Y extent of the twisted box");
  TwistedBoxYLengthCmd->SetParameterName("YLength", false);
  TwistedBoxYLengthCmd->SetRange("YLength>0.");
  TwistedBoxYLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  TwistedBoxZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedBoxZLengthCmd->SetGuidance("Set Z extent of the twisted box");
  TwistedBoxZLengthCmd->SetParameterName("ZLength", false);
  TwistedBoxZLengthCmd->SetRange("ZLength>0.");
  TwistedBoxZLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setTwistAngle";
  TwistedBoxTwistAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedBoxTwistAngleCmd->SetGuidance("Set twist angle of the twisted box");
  TwistedBoxTwistAngleCmd->SetParameterName("TwistAngle", true);
  TwistedBoxTwistAngleCmd->SetDefaultValue(45*degree);
  TwistedBoxTwistAngleCmd->SetRange("TwistAngle<90.");
  TwistedBoxTwistAngleCmd->SetUnitCategory("Angle");
}

GateTwistedBoxMessenger::~GateTwistedBoxMessenger()
{
  delete TwistedBoxXLengthCmd;
  delete TwistedBoxYLengthCmd;
  delete TwistedBoxZLengthCmd;
  delete TwistedBoxTwistAngleCmd;
}

void GateTwistedBoxMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == TwistedBoxXLengthCmd) {
      GetTwistedBoxCreator()->SetTwistedBoxXLength(TwistedBoxXLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedBoxYLengthCmd) {
      GetTwistedBoxCreator()->SetTwistedBoxYLength(TwistedBoxYLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedBoxZLengthCmd) {
      GetTwistedBoxCreator()->SetTwistedBoxZLength(TwistedBoxZLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedBoxTwistAngleCmd) {
      GetTwistedBoxCreator()->SetTwistedBoxTwistAngle(TwistedBoxTwistAngleCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
