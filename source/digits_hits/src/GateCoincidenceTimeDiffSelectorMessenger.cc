/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceTimeDiffSelectorMessenger.hh"

#include "GateCoincidenceTimeDiffSelector.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

GateCoincidenceTimeDiffSelectorMessenger::GateCoincidenceTimeDiffSelectorMessenger(GateCoincidenceTimeDiffSelector* itsTimeDiffSelector)
    : GateClockDependentMessenger(itsTimeDiffSelector)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setMin";
  minTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  minTimeCmd->SetGuidance("Set min time diff to be accepted (negative values : any is time diff is ok)");
  minTimeCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "setMax";
  maxTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxTimeCmd->SetGuidance("Set max time diff to be accepted (negative values : any is time diff is ok)");
  maxTimeCmd->SetUnitCategory("Time");


}


GateCoincidenceTimeDiffSelectorMessenger::~GateCoincidenceTimeDiffSelectorMessenger()
{
  delete minTimeCmd;
  delete minTimeCmd;
}


void GateCoincidenceTimeDiffSelectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command==minTimeCmd)
    GetTimeDiffSelector()->SetMinTime(minTimeCmd->GetNewDoubleValue(newValue));
  else if (command == maxTimeCmd)
    GetTimeDiffSelector()->SetMaxTime(maxTimeCmd->GetNewDoubleValue(newValue));
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
