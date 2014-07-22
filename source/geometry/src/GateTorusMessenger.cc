/*
 * GateTorusMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateTorusMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateTorus.hh"

GateTorusMessenger::GateTorusMessenger(GateTorus* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setInnerR";
  TorusInnerRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TorusInnerRCmd->SetGuidance("Set inner radius of the torus");
  TorusInnerRCmd->SetParameterName("InnerR", false);
  TorusInnerRCmd->SetUnitCategory("Length");

  cmdName = dir + "setOuterR";
  TorusOuterRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TorusOuterRCmd->SetGuidance("Set outer radius of the torus");
  TorusOuterRCmd->SetParameterName("OuterR", false);
  TorusOuterRCmd->SetRange("OuterR>0.");
  TorusOuterRCmd->SetUnitCategory("Length");

  cmdName = dir + "setStartPhi";
  TorusStartPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TorusStartPhiCmd->SetGuidance("Set start angle of the torus segment");
  TorusStartPhiCmd->SetParameterName("StartPhi", true);
  TorusStartPhiCmd->SetDefaultValue(0*degree);
  TorusStartPhiCmd->SetUnitCategory("Angle");

  cmdName = dir + "setDeltaPhi";
  TorusDeltaPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TorusDeltaPhiCmd->SetGuidance("Set delta angle of the torus segment");
  TorusDeltaPhiCmd->SetParameterName("DeltaPhi", true);
  TorusDeltaPhiCmd->SetDefaultValue(360*degree);
  TorusDeltaPhiCmd->SetUnitCategory("Angle");

  cmdName = dir + "setTorusR";
  TorusTorusRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  TorusTorusRCmd->SetGuidance("Set toroidal radius of the torus");
  TorusTorusRCmd->SetParameterName("TorusR", false);
  TorusTorusRCmd->SetUnitCategory("Length");
}

GateTorusMessenger::~GateTorusMessenger()
{
  delete TorusInnerRCmd;
  delete TorusOuterRCmd;
  delete TorusStartPhiCmd;
  delete TorusDeltaPhiCmd;
  delete TorusTorusRCmd;
}

void GateTorusMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == TorusInnerRCmd ) {
      GetTorusCreator()->SetTorusInnerR(TorusInnerRCmd->GetNewDoubleValue(newValue));
  } else if (command == TorusOuterRCmd) {
      GetTorusCreator()->SetTorusOuterR(TorusOuterRCmd->GetNewDoubleValue(newValue));
  } else if (command == TorusTorusRCmd) {
      GetTorusCreator()->SetTorusTorusR(TorusTorusRCmd->GetNewDoubleValue(newValue));
  } else if (command == TorusStartPhiCmd) {
      GetTorusCreator()->SetTorusStartPhi(TorusStartPhiCmd->GetNewDoubleValue(newValue));
  } else if (command == TorusDeltaPhiCmd) {
      GetTorusCreator()->SetTorusDeltaPhi(TorusDeltaPhiCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
