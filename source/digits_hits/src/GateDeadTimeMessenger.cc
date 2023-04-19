/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GateDeadTimeMessenger
    \brief  Messenger for the GateDeadTime

    - GateDeadTime - by Luc.Simon@iphe.unil.ch

    \sa GateDeadTime, GateDeadTimeMessenger
*/



#include "GateDeadTimeMessenger.hh"
#include "GateDeadTime.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIdirectory.hh"



GateDeadTimeMessenger::GateDeadTimeMessenger (GateDeadTime* DeadTime)
:GateClockDependentMessenger(DeadTime),
 	 m_DeadTime(DeadTime)
{
	G4String guidance;
	G4String cmdName;

  cmdName = GetDirectoryName() + "setDeadTime";
  DeadTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  DeadTimeCmd->SetGuidance("Set Dead time (in ps) for pulse-discrimination");
  DeadTimeCmd->SetUnitCategory("Time");

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
	  delete DeadTimeCmd;
	  delete newVolCmd;
	  delete modeCmd;
	  delete bufferSizeCmd;
	  delete bufferModeCmd;
}


void GateDeadTimeMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{
	if (aCommand== DeadTimeCmd)
	    { m_DeadTime->SetDeadTime(DeadTimeCmd->GetNewDoubleValue(newValue)); }
	  else if (aCommand==newVolCmd )
	    m_DeadTime->CheckVolumeName(newValue);
	  else if (aCommand == modeCmd)
	    m_DeadTime->SetDeadTimeMode(newValue);
	  else if (aCommand == bufferModeCmd)
	    m_DeadTime->SetBufferMode(bufferModeCmd->GetNewIntValue(newValue));
	  else if (aCommand == bufferSizeCmd)
	    m_DeadTime->SetBufferSize(bufferSizeCmd->GetNewDoubleValue(newValue));
	  else
	  {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













