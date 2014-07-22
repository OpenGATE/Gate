/*
 * GateCutTubsMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateCutTubsMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"

#include "GateCutTubs.hh"

GateCutTubsMessenger::GateCutTubsMessenger(GateCutTubs* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setInnerR";
  CutTubsInnerRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  CutTubsInnerRCmd->SetGuidance("Set inner radius of the cutted tube");
  CutTubsInnerRCmd->SetParameterName("InnerR", false);
  CutTubsInnerRCmd->SetUnitCategory("Length");

  cmdName = dir + "setOuterR";
  CutTubsOuterRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  CutTubsOuterRCmd->SetGuidance("Set outer radius of the cutted tube");
  CutTubsOuterRCmd->SetParameterName("OuterR", false);
  CutTubsOuterRCmd->SetRange("OuterR>0.");
  CutTubsOuterRCmd->SetUnitCategory("Length");

  cmdName = dir + "setStartPhi";
  CutTubsStartPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  CutTubsStartPhiCmd->SetGuidance("Set start angle of the  cutted tube segment");
  CutTubsStartPhiCmd->SetParameterName("StartPhi", true);
  CutTubsStartPhiCmd->SetDefaultValue(0*degree);
  CutTubsStartPhiCmd->SetUnitCategory("Angle");

  cmdName = dir + "setDeltaPhi";
  CutTubsDeltaPhiCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  CutTubsDeltaPhiCmd->SetGuidance("Set delta angle of the cutted tube segment");
  CutTubsDeltaPhiCmd->SetParameterName("DeltaPhi", true);
  CutTubsDeltaPhiCmd->SetDefaultValue(360*degree);
  CutTubsDeltaPhiCmd->SetUnitCategory("Angle");

  cmdName = dir + "setZLength";
  CutTubsZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  CutTubsZLengthCmd->SetGuidance("Set length radius of the cutted tube");
  CutTubsZLengthCmd->SetParameterName("ZLength", false);
  CutTubsZLengthCmd->SetUnitCategory("Length");

  cmdName = dir + "setNegativeNorm";
  CutTubsNegNormCmd = new G4UIcmdWith3Vector(cmdName.c_str(), this);
  CutTubsNegNormCmd->SetGuidance("Set cutting plane normal at -z/2");
  CutTubsNegNormCmd->SetParameterName("NegativeNormX", "NegativeNormY", "NegativeNormZ", true);
  CutTubsNegNormCmd->SetDefaultValue(G4ThreeVector(0,0,-1));

  cmdName = dir + "setPositiveNorm";
  CutTubsPosNormCmd = new G4UIcmdWith3Vector(cmdName.c_str(), this);
  CutTubsPosNormCmd->SetGuidance("Set cutting plane normal at +z/2");
  CutTubsPosNormCmd->SetParameterName("PositiveNormX", "PositiveNormY", "PositiveNormZ", true);
  CutTubsPosNormCmd->SetDefaultValue(G4ThreeVector(0,0,1));
}

GateCutTubsMessenger::~GateCutTubsMessenger()
{
  delete CutTubsInnerRCmd;
  delete CutTubsOuterRCmd;
  delete CutTubsStartPhiCmd;
  delete CutTubsDeltaPhiCmd;
  delete CutTubsZLengthCmd;
  delete CutTubsNegNormCmd;
  delete CutTubsPosNormCmd;
}

void GateCutTubsMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == CutTubsInnerRCmd ) {
      GetCutTubsCreator()->SetCutTubsInnerR(CutTubsInnerRCmd->GetNewDoubleValue(newValue));
  } else if (command == CutTubsOuterRCmd) {
      GetCutTubsCreator()->SetCutTubsOuterR(CutTubsOuterRCmd->GetNewDoubleValue(newValue));
  } else if (command == CutTubsZLengthCmd) {
      GetCutTubsCreator()->SetCutTubsZLength(CutTubsZLengthCmd->GetNewDoubleValue(newValue));
  } else if (command == CutTubsStartPhiCmd) {
      GetCutTubsCreator()->SetCutTubsStartPhi(CutTubsStartPhiCmd->GetNewDoubleValue(newValue));
  } else if (command == CutTubsDeltaPhiCmd) {
      GetCutTubsCreator()->SetCutTubsDeltaPhi(CutTubsDeltaPhiCmd->GetNewDoubleValue(newValue));
  } else if (command == CutTubsNegNormCmd) {
        GetCutTubsCreator()->SetCutTubsNegNorm(CutTubsNegNormCmd->GetNew3VectorValue(newValue));
  } else if (command == CutTubsPosNormCmd) {
        GetCutTubsCreator()->SetCutTubsPosNorm(CutTubsPosNormCmd->GetNew3VectorValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
