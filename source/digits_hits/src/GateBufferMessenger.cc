/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBufferMessenger.hh"
#include "GateBuffer.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include <G4UIcmdWithAnInteger.hh>
#include <G4UIcmdWithABool.hh>

GateBufferMessenger::GateBufferMessenger(GateBuffer* itsPulseProcessor)
    : GatePulseProcessorMessenger(itsPulseProcessor)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setBufferSize";
  m_bufferSizeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  m_bufferSizeCmd->SetGuidance("Set the buffer size");
  m_bufferSizeCmd->SetUnitCategory("Memory size");

  cmdName = GetDirectoryName() + "setReadFrequency";
  m_readFrequencyCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  m_readFrequencyCmd->SetGuidance("set the buffer read frequency");
  m_readFrequencyCmd->SetUnitCategory("Frequency");

  cmdName = GetDirectoryName() + "modifyTime";
  m_modifyTimeCmd = new G4UIcmdWithABool(cmdName,this);
  m_modifyTimeCmd->SetGuidance("does the buffer modify the time of pulses");

  cmdName = GetDirectoryName() + "setDepth";
  m_setDepthCmd = new G4UIcmdWithAnInteger(cmdName,this);
  m_setDepthCmd->SetGuidance("the depth of each individual buffer");

  cmdName = GetDirectoryName() + "setMode";
  m_setModeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  m_setModeCmd->SetGuidance("How the buffer is read");
  m_setModeCmd->SetParameterName("mode",false);
  m_setModeCmd->SetRange("0<=mode<=1");
}


GateBufferMessenger::~GateBufferMessenger()
{
  delete m_bufferSizeCmd;
  delete m_readFrequencyCmd;
  delete m_modifyTimeCmd;
  delete m_setDepthCmd;
  delete m_setModeCmd;
}


void GateBufferMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateBuffer* Buffer = static_cast<GateBuffer*>(GetNamedObject());
  if (!Buffer) return;
  if (command== m_bufferSizeCmd)
    { Buffer->SetBufferSize((long long unsigned int) m_bufferSizeCmd->GetNewDoubleValue(newValue)); }
  else if (command == m_readFrequencyCmd)
    Buffer->SetReadFrequency(m_readFrequencyCmd->GetNewDoubleValue(newValue));
  else if (command == m_modifyTimeCmd)
    Buffer->SetDoModifyTime(m_modifyTimeCmd->GetNewBoolValue(newValue));
  else if (command == m_setDepthCmd)
    Buffer->SetDepth(m_setDepthCmd->GetNewIntValue(newValue));
  else if (command == m_setModeCmd)
    Buffer->SetMode(m_setModeCmd->GetNewIntValue(newValue));
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
