/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateDeadTimeMessenger.hh"
#include "GateDeadTime.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

GateDeadTimeMessenger::GateDeadTimeMessenger(GateDeadTime* itsDeadTime)
  : GatePulseProcessorMessenger(itsDeadTime)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setDeadTime";
  deadTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  deadTimeCmd->SetGuidance("Set Dead time (in ps) for pulse-discrimination");
  deadTimeCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "chooseDTVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume (depth) for dead time(e.g. crystal)");

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

}


GateDeadTimeMessenger::~GateDeadTimeMessenger()
{
  delete deadTimeCmd;
  delete newVolCmd;
  delete modeCmd;
  delete bufferSizeCmd;
  delete bufferModeCmd;
}


void GateDeadTimeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command== deadTimeCmd)
    { GetDeadTime()->SetDeadTime(deadTimeCmd->GetNewDoubleValue(newValue)); }
  else if (command==newVolCmd )
    GetDeadTime()->CheckVolumeName(newValue);
  else if (command == modeCmd)
    GetDeadTime()->SetDeadTimeMode(newValue);
  else if (command == bufferModeCmd)
    GetDeadTime()->SetBufferMode(bufferModeCmd->GetNewIntValue(newValue));
  else if (command == bufferSizeCmd)
    GetDeadTime()->SetBufferSize(bufferSizeCmd->GetNewDoubleValue(newValue));
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
