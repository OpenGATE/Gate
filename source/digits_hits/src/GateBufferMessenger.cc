/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

#include "GateBufferMessenger.hh"
#include "GateBuffer.hh"
#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIdirectory.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"

GateBufferMessenger::GateBufferMessenger (GateBuffer* Buffer)
:GateClockDependentMessenger(Buffer),
 	 m_Buffer(Buffer)
{
	G4String guidance;
	G4String cmdName;

    cmdName = GetDirectoryName() + "setBufferSize";
    m_BufferSizeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
    m_BufferSizeCmd->SetGuidance("Set the Buffer size");
    m_BufferSizeCmd->SetUnitCategory("Memory size");

    cmdName = GetDirectoryName() + "setReadFrequency";
    m_readFrequencyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
    m_readFrequencyCmd->SetGuidance("set the Buffer read frequency");
    m_readFrequencyCmd->SetUnitCategory("Frequency");

    cmdName = GetDirectoryName() + "modifyTime";
    m_modifyTimeCmd = new G4UIcmdWithABool(cmdName,this);
    m_modifyTimeCmd->SetGuidance("does the Buffer modify the time of pulses");

    cmdName = GetDirectoryName() + "setDepth";
    m_setDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
    m_setDepthCmd->SetGuidance("the depth of each individual Buffer");

    cmdName = GetDirectoryName() + "setMode";
    m_setModeCmd = new G4UIcmdWithAnInteger(cmdName,this);
    m_setModeCmd->SetGuidance("How the Buffer is read");
    m_setModeCmd->SetParameterName("mode",false);
    m_setModeCmd->SetRange("0<=mode<=1");



}


GateBufferMessenger::~GateBufferMessenger()
{
	  delete m_BufferSizeCmd;
	  delete m_readFrequencyCmd;
	  delete m_modifyTimeCmd;
	  delete m_setDepthCmd;
	  delete m_setModeCmd;
}


void GateBufferMessenger::SetNewValue(G4UIcommand * aCommand,G4String newValue)
{

	if (aCommand == m_BufferSizeCmd)
	      {
			m_Buffer->SetBufferSize((long long unsigned int) m_BufferSizeCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand == m_readFrequencyCmd)
	      {
			m_Buffer->SetReadFrequency(m_readFrequencyCmd->GetNewDoubleValue(newValue));
	      }
	else if (aCommand == m_modifyTimeCmd)
	      {
			m_Buffer->SetDoModifyTime(m_modifyTimeCmd->GetNewBoolValue(newValue));
	      }
	else if (aCommand == m_setDepthCmd)
	      {
			m_Buffer->SetDepth(m_setDepthCmd->GetNewIntValue(newValue));
	      }
	else if (aCommand == m_setModeCmd)
	      {
			m_Buffer->SetMode(m_setModeCmd->GetNewIntValue(newValue));
	      }
	else
	    {
	    	GateClockDependentMessenger::SetNewValue(aCommand,newValue);
	    }
}













