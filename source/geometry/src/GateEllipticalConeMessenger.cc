/*
 * GateEllipticalConeMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateEllipticalConeMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateEllipticalCone.hh"

GateEllipticalConeMessenger::GateEllipticalConeMessenger(GateEllipticalCone* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setXSemiaxis";
  EllipticalConeXSemiAxisCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipticalConeXSemiAxisCmd->SetGuidance("Set x semiaxis at the bottom of the elliptical cone");
  EllipticalConeXSemiAxisCmd->SetParameterName("XSemiaxis", false);
  EllipticalConeXSemiAxisCmd->SetUnitCategory("Length");

  cmdName = dir + "setYSemiaxis";
  EllipticalConeYSemiAxisCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipticalConeYSemiAxisCmd->SetGuidance("Set y semiaxis at the bottom of the elliptical cone");
  EllipticalConeYSemiAxisCmd->SetParameterName("YSemiaxis", false);
  EllipticalConeYSemiAxisCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  EllipticalConeZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipticalConeZLengthCmd->SetGuidance("Set total height of the elliptical cone");
  EllipticalConeZLengthCmd->SetParameterName("ZLength", false);
  EllipticalConeZLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setZCut";
  EllipticalConeZCutCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  EllipticalConeZCutCmd->SetGuidance("Set cutted height of the elliptical cone");
  EllipticalConeZCutCmd->SetParameterName("ZCut", true);
  EllipticalConeZCutCmd->SetDefaultValue(GetEllipticalConeCreator()->GetEllipticalConeZLength());
  EllipticalConeZCutCmd->SetUnitCategory("Length");
}

GateEllipticalConeMessenger::~GateEllipticalConeMessenger()
{
  delete EllipticalConeXSemiAxisCmd;
  delete EllipticalConeYSemiAxisCmd;
  delete EllipticalConeZLengthCmd;
  delete EllipticalConeZCutCmd;
}

void GateEllipticalConeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == EllipticalConeXSemiAxisCmd ) {
      GetEllipticalConeCreator()->SetEllipticalConeXSemiAxis(EllipticalConeXSemiAxisCmd->GetNewDoubleValue(newValue));
  } else if (command == EllipticalConeYSemiAxisCmd) {
      GetEllipticalConeCreator()->SetEllipticalConeYSemiAxis(EllipticalConeYSemiAxisCmd->GetNewDoubleValue(newValue));
  } else if (command == EllipticalConeZLengthCmd) {
      GetEllipticalConeCreator()->SetEllipticalConeZLength(EllipticalConeZLengthCmd->GetNewDoubleValue(newValue));
} else if (command == EllipticalConeZCutCmd) {
    GetEllipticalConeCreator()->SetEllipticalConeZCut(EllipticalConeZCutCmd->GetNewDoubleValue(newValue));
} else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
