/*
 * GateTwistedTubsMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTwistedTubsMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateTwistedTubs.hh"

GateTwistedTubsMessenger::GateTwistedTubsMessenger(GateTwistedTubs* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setInnerR";
  TwistedTubsInnerRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTubsInnerRCmd->SetGuidance("Set inner radius of the twisted tubs");
  TwistedTubsInnerRCmd->SetParameterName("InnerR", false);
  TwistedTubsInnerRCmd->SetUnitCategory("Length");

  cmdName = dir + "setOuterR";
  TwistedTubsOuterRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTubsOuterRCmd->SetGuidance("Set outer radius of the twisted tubs");
  TwistedTubsOuterRCmd->SetParameterName("OuterR", false);
  TwistedTubsOuterRCmd->SetRange("OuterR>0.");
  TwistedTubsOuterRCmd->SetUnitCategory("Length");

  cmdName = dir + "setPositiveZ";
  TwistedTubsPosZCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTubsPosZCmd->SetGuidance("Set Z extent of the twisted tubs");
  TwistedTubsPosZCmd->SetParameterName("PosZ", false);
  TwistedTubsPosZCmd->SetUnitCategory("Length");

  cmdName = dir + "setNegativeZ";
  TwistedTubsNegZCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTubsNegZCmd->SetGuidance("Set Z extent of the twisted tubs");
  TwistedTubsNegZCmd->SetParameterName("NegZ", false);
  TwistedTubsNegZCmd->SetUnitCategory("Length");

  cmdName = dir + "setTotalPhi";
  TwistedTubsTotalPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTubsTotalPhiCmd->SetGuidance("Set total phi coverage of the twisted tubs");
  TwistedTubsTotalPhiCmd->SetParameterName("TotalPhi", true);
  TwistedTubsTotalPhiCmd->SetDefaultValue(90*degree);
  TwistedTubsTotalPhiCmd->SetUnitCategory("Angle");

  cmdName = dir + "setTwistAngle";
  TwistedTubsTwistAngleCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TwistedTubsTwistAngleCmd->SetGuidance("Set twist angle of the twisted tubs");
  TwistedTubsTwistAngleCmd->SetParameterName("TwistAngle", true);
  TwistedTubsTwistAngleCmd->SetRange("TwistAngle<90.");
  TwistedTubsTwistAngleCmd->SetDefaultValue(45*degree);
  TwistedTubsTwistAngleCmd->SetUnitCategory("Angle");

  cmdName = dir + "setNSegment";
  TwistedTubsNSegmentCmd = new G4UIcmdWithAnInteger(cmdName.c_str(), this);
  TwistedTubsNSegmentCmd->SetGuidance("Set number of segments of the twisted tubs");
  TwistedTubsNSegmentCmd->SetParameterName("NSegment", true);
  TwistedTubsNSegmentCmd->SetDefaultValue(1);
}

GateTwistedTubsMessenger::~GateTwistedTubsMessenger()
{
  delete TwistedTubsInnerRCmd;
  delete TwistedTubsOuterRCmd;
  delete TwistedTubsPosZCmd;
  delete TwistedTubsNegZCmd;
  delete TwistedTubsTotalPhiCmd;
  delete TwistedTubsTwistAngleCmd;
  delete TwistedTubsNSegmentCmd;
}

void GateTwistedTubsMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == TwistedTubsInnerRCmd ) {
      GetTwistedTubsCreator()->SetTwistedTubsInnerR(TwistedTubsInnerRCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTubsOuterRCmd) {
      GetTwistedTubsCreator()->SetTwistedTubsOuterR(TwistedTubsOuterRCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTubsPosZCmd) {
      GetTwistedTubsCreator()->SetTwistedTubsPosZ(TwistedTubsPosZCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTubsNegZCmd) {
      GetTwistedTubsCreator()->SetTwistedTubsNegZ(TwistedTubsNegZCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTubsTotalPhiCmd) {
      GetTwistedTubsCreator()->SetTwistedTubsTotalPhi(TwistedTubsTotalPhiCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTubsTwistAngleCmd) {
      GetTwistedTubsCreator()->SetTwistedTubsTwistAngle(TwistedTubsTwistAngleCmd->GetNewDoubleValue(newValue));
  } else if (command == TwistedTubsNSegmentCmd) {
      GetTwistedTubsCreator()->SetTwistedTubsNSegment(TwistedTubsNSegmentCmd->GetNewIntValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
