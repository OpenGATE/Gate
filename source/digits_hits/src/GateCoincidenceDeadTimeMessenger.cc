/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

#include "GateCoincidenceDeadTimeMessenger.hh"
#include "GateCoincidenceDeadTime.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIdirectory.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"


GateCoincidenceDeadTimeMessenger::GateCoincidenceDeadTimeMessenger (GateCoincidenceDeadTime* CoincidenceDeadTime)
:GateClockDependentMessenger(CoincidenceDeadTime),
 	 m_CoincidenceDeadTime(CoincidenceDeadTime)
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


void GateCoincidenceDeadTimeMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if (aCommand== deadTimeCmd)
	    { m_CoincidenceDeadTime->SetDeadTime(deadTimeCmd->GetNewDoubleValue(newValue)); }
	  else if (aCommand == modeCmd)
	    m_CoincidenceDeadTime->SetDeadTimeMode(newValue);
	  else if (aCommand == bufferModeCmd)
	    m_CoincidenceDeadTime->SetBufferMode(bufferModeCmd->GetNewIntValue(newValue));
	  else if (aCommand == bufferSizeCmd)
	    m_CoincidenceDeadTime->SetBufferSize(bufferSizeCmd->GetNewDoubleValue(newValue));
	  else if (aCommand == conserveAllEventCmd)
	    m_CoincidenceDeadTime->SetConserveAllEvent(conserveAllEventCmd->GetNewBoolValue(newValue));
	  else
	    GateClockDependentMessenger::SetNewValue(aCommand,newValue);

}













