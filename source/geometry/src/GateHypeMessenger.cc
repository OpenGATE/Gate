/*
 * GateHypeMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateHypeMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateHype.hh"

GateHypeMessenger::GateHypeMessenger(GateHype* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setInnerR";
  HypeInnerRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  HypeInnerRCmd->SetGuidance("Set inner radius of the hyperbolic tube");
  HypeInnerRCmd->SetParameterName("InnerR", false);
  HypeInnerRCmd->SetUnitCategory("Length");

  cmdName = dir + "setOuterR";
  HypeOuterRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  HypeOuterRCmd->SetGuidance("Set outer radius of the hyperbolic tube");
  HypeOuterRCmd->SetParameterName("OuterR", false);
  HypeOuterRCmd->SetRange("OuterR>0.");
  HypeOuterRCmd->SetUnitCategory("Length");

  cmdName = dir + "setInnerStereo";
  HypeInnerStereoCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  HypeInnerStereoCmd->SetGuidance("Set inner stereo angle of the hyperbolic tube");
  HypeInnerStereoCmd->SetParameterName("InnerStereo", true);
  HypeInnerStereoCmd->SetDefaultValue(0*degree);
  HypeInnerStereoCmd->SetUnitCategory("Angle");

  cmdName = dir + "setOuterStereo";
  HypeOuterStereoCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  HypeOuterStereoCmd->SetGuidance("Set outer stereo angle of the hyperbolic tube");
  HypeOuterStereoCmd->SetParameterName("OuterStereo", true);
  HypeOuterStereoCmd->SetDefaultValue(0*degree);
  HypeOuterStereoCmd->SetUnitCategory("Angle");

  cmdName = dir + "setZLength";
  HypeZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  HypeZLengthCmd->SetGuidance("Set Z extent of the hyperbolic tube");
  HypeZLengthCmd->SetParameterName("ZLength", false);
  HypeZLengthCmd->SetUnitCategory("Length");
}

GateHypeMessenger::~GateHypeMessenger()
{
  delete HypeInnerRCmd;
  delete HypeOuterRCmd;
  delete HypeInnerStereoCmd;
  delete HypeOuterStereoCmd;
  delete HypeZLengthCmd;
}

void GateHypeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == HypeInnerRCmd ) {
      GetHypeCreator()->SetHypeInnerR(HypeInnerRCmd->GetNewDoubleValue(newValue));
  } else if (command == HypeOuterRCmd) {
      GetHypeCreator()->SetHypeOuterR(HypeOuterRCmd->GetNewDoubleValue(newValue));
  } else if (command == HypeZLengthCmd) {
      GetHypeCreator()->SetHypeZLength(HypeZLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == HypeInnerStereoCmd) {
      GetHypeCreator()->SetHypeInnerStereo(HypeInnerStereoCmd->GetNewDoubleValue(newValue));
  } else if (command == HypeOuterStereoCmd) {
      GetHypeCreator()->SetHypeOuterStereo(HypeOuterStereoCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
