/*
 * GateTwistedTrdMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedTrdMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateTwistedTrd.hh"

GateTwistedTrdMessenger::GateTwistedTrdMessenger(GateTwistedTrd* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setX1Length";
  TwistedTrdX1LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrdX1LengthCmd->SetGuidance("Set X1 length of the twisted trapezoid");
  TwistedTrdX1LengthCmd->SetParameterName("X1Length", false);
  TwistedTrdX1LengthCmd->SetRange("X1Length>0.");
  TwistedTrdX1LengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setX2Length";
  TwistedTrdX2LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrdX2LengthCmd->SetGuidance("Set X2 length of the twisted trapezoid");
  TwistedTrdX2LengthCmd->SetParameterName("X2Length", false);
  TwistedTrdX2LengthCmd->SetRange("X2Length>0.");
  TwistedTrdX2LengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setY1Length";
  TwistedTrdY1LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrdY1LengthCmd->SetGuidance("Set Y1 length of the twisted trapezoid");
  TwistedTrdY1LengthCmd->SetParameterName("Y1Length", false);
  TwistedTrdY1LengthCmd->SetRange("Y1Length>0.");
  TwistedTrdY1LengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setY2Length";
  TwistedTrdY2LengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrdY2LengthCmd->SetGuidance("Set Y2 length of the twisted trapezoid");
  TwistedTrdY2LengthCmd->SetParameterName("Y2Length", false);
  TwistedTrdY2LengthCmd->SetRange("Y2Length>0.");
  TwistedTrdY2LengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  TwistedTrdZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrdZLengthCmd->SetGuidance("Set Z extent of the twisted trapezoid");
  TwistedTrdZLengthCmd->SetParameterName("ZLength", false);
  TwistedTrdZLengthCmd->SetRange("ZLength>0.");
  TwistedTrdZLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setTwistAngle";
  TwistedTrdTwistAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrdTwistAngleCmd->SetGuidance("Set twist angle of the twisted trapezoid");
  TwistedTrdTwistAngleCmd->SetParameterName("TwistAngle", true);
  TwistedTrdTwistAngleCmd->SetDefaultValue(45*degree);
  TwistedTrdTwistAngleCmd->SetRange("TwistAngle<90.");
  TwistedTrdTwistAngleCmd->SetUnitCategory("Angle");
}

GateTwistedTrdMessenger::~GateTwistedTrdMessenger()
{
  delete TwistedTrdX1LengthCmd;
  delete TwistedTrdX2LengthCmd;
  delete TwistedTrdY1LengthCmd;
  delete TwistedTrdY2LengthCmd;
  delete TwistedTrdZLengthCmd;
  delete TwistedTrdTwistAngleCmd;
}

void GateTwistedTrdMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == TwistedTrdX1LengthCmd) {
      GetTwistedTrdCreator()->SetTwistedTrdX1Length(TwistedTrdX1LengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrdX2LengthCmd) {
      GetTwistedTrdCreator()->SetTwistedTrdX2Length(TwistedTrdX2LengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrdY1LengthCmd) {
      GetTwistedTrdCreator()->SetTwistedTrdY1Length(TwistedTrdY1LengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrdY2LengthCmd) {
      GetTwistedTrdCreator()->SetTwistedTrdY2Length(TwistedTrdY2LengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrdZLengthCmd) {
      GetTwistedTrdCreator()->SetTwistedTrdZLength(TwistedTrdZLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrdTwistAngleCmd) {
      GetTwistedTrdCreator()->SetTwistedTrdTwistAngle(TwistedTrdTwistAngleCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
