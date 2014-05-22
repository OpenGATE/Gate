/*----------------------
  03/2012
  ----------------------*/


#include "GateTriCoincidenceSorterMessenger.hh"

#include "GateTriCoincidenceSorter.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"

GateTriCoincidenceSorterMessenger::GateTriCoincidenceSorterMessenger(GateTriCoincidenceSorter* itsProcessor)
   : GateClockDependentMessenger(itsProcessor)
{
   m_itsProcessor = itsProcessor;
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName()+"setSinglesPulseListName";
  m_SetInputSPLNameCmd = new G4UIcmdWithAString(cmdName,this);
  m_SetInputSPLNameCmd->SetGuidance("Add a name for the singles input pulse channel");
  m_SetInputSPLNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName() + "setWindow";
  m_triCoincWindowCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  m_triCoincWindowCmd->SetGuidance("Set the time-window for coincidence between selected singles and the PET coincidence pulse");
  m_triCoincWindowCmd->SetUnitCategory("Time");
  m_triCoincWindowCmd->SetParameterName("window",false);
  m_triCoincWindowCmd->SetRange("0<=window<=100");

  cmdName = GetDirectoryName() + "setSinglesBufferSize";
  m_SetWSPulseListSizeCmd = new G4UIcmdWithAnInteger(cmdName,this);
  m_SetWSPulseListSizeCmd->SetGuidance("Set the buffer size of waiting singles");
  m_SetWSPulseListSizeCmd->SetParameterName("size",false);
  m_SetWSPulseListSizeCmd->SetRange("0<=size<=1000");
}


GateTriCoincidenceSorterMessenger::~GateTriCoincidenceSorterMessenger()
{
   delete m_SetInputSPLNameCmd;
   delete m_triCoincWindowCmd;
   delete m_SetWSPulseListSizeCmd;
}


void GateTriCoincidenceSorterMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
   if (command == m_SetInputSPLNameCmd)
   { m_itsProcessor->SetSinglesPulseListName(newValue); }

  else if (command == m_triCoincWindowCmd)
  {m_itsProcessor->SetTriCoincWindow(m_triCoincWindowCmd->GetNewDoubleValue(newValue));}

  else if (command == m_SetWSPulseListSizeCmd)
  {m_itsProcessor->SetWSPulseListSize(m_SetWSPulseListSizeCmd->GetNewIntValue(newValue));}

  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
