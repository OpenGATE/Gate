/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceDeadTimeMessenger.hh"

#include "GateCoincidenceDeadTime.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"

GateCoincidenceDeadTimeMessenger::GateCoincidenceDeadTimeMessenger(GateCoincidenceDeadTime* itsDeadTime)
    : GateClockDependentMessenger(itsDeadTime)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setDeadTime";
  deadTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  deadTimeCmd->SetGuidance("Set Dead time (in ps) for pulse-discrimination");
  deadTimeCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "setMode";
  modeCmd = new G4UIcmdWithAString(cmdName,this);
  modeCmd->SetGuidance("set a mode for dead time");
  modeCmd->SetGuidance("paralysable nonparalysable");

  cmdName = GetDirectoryName() + "setBufferMode";
  bufferModeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  bufferModeCmd->SetGuidance("set a mode for buffer management");
  bufferModeCmd->SetGuidance("0 : DT during writing, 1 : DT if writing AND buffer full");

  cmdName = GetDirectoryName() + "setBufferSize";
  bufferSizeCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  bufferSizeCmd->SetGuidance("set the buffer size");
  bufferSizeCmd->SetUnitCategory("Memory size");

  cmdName = GetDirectoryName() + "conserveAllEvent";
  conserveAllEventCmd = new G4UIcmdWithABool(cmdName,this);
  conserveAllEventCmd->SetGuidance("True if an event is kept or killed entierly");

}


GateCoincidenceDeadTimeMessenger::~GateCoincidenceDeadTimeMessenger()
{
  delete bufferSizeCmd;
  delete bufferModeCmd;
  delete deadTimeCmd;
  delete modeCmd;
  delete conserveAllEventCmd;
}


void GateCoincidenceDeadTimeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command== deadTimeCmd)
    { GetDeadTime()->SetDeadTime(deadTimeCmd->GetNewDoubleValue(newValue)); }
  else if (command == modeCmd)
    GetDeadTime()->SetDeadTimeMode(newValue);
  else if (command == bufferModeCmd)
    GetDeadTime()->SetBufferMode(bufferModeCmd->GetNewIntValue(newValue));
  else if (command == bufferSizeCmd)
    GetDeadTime()->SetBufferSize(bufferSizeCmd->GetNewDoubleValue(newValue));
  else if (command == conserveAllEventCmd)
    GetDeadTime()->SetConserveAllEvent(conserveAllEventCmd->GetNewBoolValue(newValue));
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
