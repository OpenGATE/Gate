/*
 * GateTwistedTrapMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedTrapMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

#include "GateTwistedTrap.hh"

GateTwistedTrapMessenger::GateTwistedTrapMessenger(GateTwistedTrap* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setYMinusLength";
  TwistedTrapYMinusLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapYMinusLengthCmd->SetGuidance("Set y length at -z/2 of the twisted trapezoid");
  TwistedTrapYMinusLengthCmd->SetParameterName("YMinusLength", false);
  TwistedTrapYMinusLengthCmd->SetRange("YMinusLength>0.");
  TwistedTrapYMinusLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setYPlusLength";
  TwistedTrapYPlusLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapYPlusLengthCmd->SetGuidance("Set y length at +z/2 of the twisted trapezoid");
  TwistedTrapYPlusLengthCmd->SetParameterName("YPlusLength", false);
  TwistedTrapYPlusLengthCmd->SetRange("YPlusLength>0.");
  TwistedTrapYPlusLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setX1PlusLength";
  TwistedTrapX1PlusLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapX1PlusLengthCmd->SetGuidance("Set x length at +z/2, -y/2 of the twisted trapezoid");
  TwistedTrapX1PlusLengthCmd->SetParameterName("X1PlusLength", false);
  TwistedTrapX1PlusLengthCmd->SetRange("X1PlusLength>0.");
  TwistedTrapX1PlusLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setX2PlusLength";
  TwistedTrapX2PlusLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapX2PlusLengthCmd->SetGuidance("Set y length at +z/2, +y/2 of the twisted trapezoid");
  TwistedTrapX2PlusLengthCmd->SetParameterName("X2PlusLength", false);
  TwistedTrapX2PlusLengthCmd->SetRange("X2PlusLength>0.");
  TwistedTrapX2PlusLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setX1MinusLength";
  TwistedTrapX1MinusLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapX1MinusLengthCmd->SetGuidance("Set x length at -z/2, -y/2 of the twisted trapezoid");
  TwistedTrapX1MinusLengthCmd->SetParameterName("X1MinusLength", false);
  TwistedTrapX1MinusLengthCmd->SetRange("X1MinusLength>0.");
  TwistedTrapX1MinusLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setX2MinusLength";
  TwistedTrapX2MinusLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapX2MinusLengthCmd->SetGuidance("Set y length at -z/2, +y/2 of the twisted trapezoid");
  TwistedTrapX2MinusLengthCmd->SetParameterName("X2MinusLength", false);
  TwistedTrapX2MinusLengthCmd->SetRange("X2MinusLength>0.");
  TwistedTrapX2MinusLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  TwistedTrapZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapZLengthCmd->SetGuidance("Set z length of the twisted trapezoid");
  TwistedTrapZLengthCmd->SetParameterName("ZLength", false);
  TwistedTrapZLengthCmd->SetRange("ZLength>0.");
  TwistedTrapZLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setTwistAngle";
  TwistedTrapTwistAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapTwistAngleCmd->SetGuidance("Set twist angle of the twisted trapezoid");
  TwistedTrapTwistAngleCmd->SetParameterName("TwistAngle", true);
  TwistedTrapTwistAngleCmd->SetDefaultValue(45*degree);
  TwistedTrapTwistAngleCmd->SetRange("TwistAngle<90.");
  TwistedTrapTwistAngleCmd->SetUnitCategory("Angle");

  cmdName = dir + "setPolarAngle";
  TwistedTrapPolarAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapPolarAngleCmd->SetGuidance("Set polar angle of the twisted trapezoid");
  TwistedTrapPolarAngleCmd->SetParameterName("PolarAngle", true);
  TwistedTrapPolarAngleCmd->SetDefaultValue(45*degree);
  TwistedTrapPolarAngleCmd->SetUnitCategory("Angle");

  cmdName = dir + "setAzimuthalAngle";
  TwistedTrapAzimuthalAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapAzimuthalAngleCmd->SetGuidance("Set azimuthal angle of the twisted trapezoid");
  TwistedTrapAzimuthalAngleCmd->SetParameterName("AzimuthalAngle", true);
  TwistedTrapAzimuthalAngleCmd->SetDefaultValue(45*degree);
  TwistedTrapAzimuthalAngleCmd->SetUnitCategory("Angle");

  cmdName = dir + "setTiltAngle";
  TwistedTrapTiltAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTrapTiltAngleCmd->SetGuidance("Set tilt angle of the twisted trapezoid");
  TwistedTrapTiltAngleCmd->SetParameterName("TiltAngle", true);
  TwistedTrapTiltAngleCmd->SetDefaultValue(45*degree);
  TwistedTrapTiltAngleCmd->SetUnitCategory("Angle");
}

GateTwistedTrapMessenger::~GateTwistedTrapMessenger()
{
  delete TwistedTrapYMinusLengthCmd;
  delete TwistedTrapYPlusLengthCmd;
  delete TwistedTrapX1MinusLengthCmd;
  delete TwistedTrapX2MinusLengthCmd;
  delete TwistedTrapX1PlusLengthCmd;
  delete TwistedTrapX2PlusLengthCmd;
  delete TwistedTrapZLengthCmd;
  delete TwistedTrapTwistAngleCmd;
  delete TwistedTrapPolarAngleCmd;
  delete TwistedTrapAzimuthalAngleCmd;
  delete TwistedTrapTiltAngleCmd;
}

void GateTwistedTrapMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == TwistedTrapYMinusLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapYMinusLength(TwistedTrapYMinusLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapYPlusLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapYPlusLength(TwistedTrapYPlusLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapX1MinusLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapX1MinusLength(TwistedTrapX1MinusLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapX2MinusLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapX2MinusLength(TwistedTrapX2MinusLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapX1PlusLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapX1PlusLength(TwistedTrapX1PlusLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapX2PlusLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapX2PlusLength(TwistedTrapX2PlusLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapZLengthCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapZLength(TwistedTrapZLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapTwistAngleCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapTwistAngle(TwistedTrapTwistAngleCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapPolarAngleCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapPolarAngle(TwistedTrapPolarAngleCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapAzimuthalAngleCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapAzimuthalAngle(TwistedTrapAzimuthalAngleCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTrapTiltAngleCmd) {
      GetTwistedTrapCreator()->SetTwistedTrapTiltAngle(TwistedTrapTiltAngleCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
