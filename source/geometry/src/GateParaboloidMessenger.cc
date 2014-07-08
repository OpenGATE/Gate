/*
 * GateParaboloidMessenger.cc
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#include "GateParaboloidMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GateParaboloid.hh"

GateParaboloidMessenger::GateParaboloidMessenger(GateParaboloid* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  G4String dir = GetDirectoryName() + "geometry/";

  G4String cmdName = dir + "setNegativeR";
  ParaboloidNegativeRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaboloidNegativeRCmd->SetGuidance("Set radius of the paraboloid at -z/2");
  ParaboloidNegativeRCmd->SetParameterName("NegativeR", false);
  ParaboloidNegativeRCmd->SetUnitCategory("Length");

  cmdName = dir + "setPositiveR";
  ParaboloidPositiveRCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaboloidPositiveRCmd->SetGuidance("Set radius of the paraboloid at +z/2");
  ParaboloidPositiveRCmd->SetParameterName("PositiveR", false);
  ParaboloidPositiveRCmd->SetRange("PositiveR>0.");
  ParaboloidPositiveRCmd->SetUnitCategory("Length");

  cmdName = dir + "setZLength";
  ParaboloidZLengthCmd = new G4UIcmdWithADoubleAndUnit(cmdName.c_str(), this);
  ParaboloidZLengthCmd->SetGuidance("Set Z extent of the paraboloidrbolic tube");
  ParaboloidZLengthCmd->SetParameterName("ZLength", false);
  ParaboloidZLengthCmd->SetUnitCategory("Length");
}

GateParaboloidMessenger::~GateParaboloidMessenger()
{
  delete ParaboloidPositiveRCmd;
  delete ParaboloidNegativeRCmd;
  delete ParaboloidZLengthCmd;
}

void GateParaboloidMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == ParaboloidPositiveRCmd ) {
      GetParaboloidCreator()->SetParaboloidPositiveR(ParaboloidPositiveRCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaboloidNegativeRCmd) {
      GetParaboloidCreator()->SetParaboloidNegativeR(ParaboloidNegativeRCmd->GetNewDoubleValue(newValue));
  } else if (command == ParaboloidZLengthCmd) {
      GetParaboloidCreator()->SetParaboloidZLength(ParaboloidZLengthCmd->GetNewDoubleValue(newValue));
  } else
    GateVolumeMessenger::SetNewValue(command,newValue);
}
