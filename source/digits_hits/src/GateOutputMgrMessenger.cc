/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateOutputMgrMessenger.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//------------------------------------------------------------------------------
GateOutputMgrMessenger::GateOutputMgrMessenger(GateOutputMgr* outputMgr)
  : GateMessenger(outputMgr->GetName()),
    m_outputMgr(outputMgr)
{

  pGateOutputMess = new G4UIdirectory("/gate/output/");
  pGateOutputMess->SetGuidance("GATE output control.");

  G4String cmdName;
  cmdName = GetDirectoryName()+"describe";
 // cmdName = baseName + outputMgr->GetObjectName() + "describe";
  DescribeCmd = new G4UIcmdWithoutParameter(cmdName,this);
  DescribeCmd->SetGuidance("List of the output manager properties");

  cmdName = GetDirectoryName()+"verbose";
//  cmdName = baseName + outputMgr->GetObjectName() + "verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set GATE output manager verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  cmdName = GetDirectoryName()+"allowNoOutput";
  AllowNoOutputCmd = new G4UIcmdWithoutParameter(cmdName,this);
  AllowNoOutputCmd->SetGuidance("Allow to launch a simulation without any output nor actor");
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
GateOutputMgrMessenger::~GateOutputMgrMessenger()
{
  delete DescribeCmd;
  delete VerboseCmd;
  delete AllowNoOutputCmd;
  delete pGateOutputMess;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
void GateOutputMgrMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if( command == VerboseCmd ) {
    m_outputMgr->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  } else if( command == DescribeCmd ) {
    m_outputMgr->Describe();
  } else if( command == AllowNoOutputCmd ) {
    m_outputMgr->AllowNoOutput();
  } else
  GateMessenger::SetNewValue(command, newValue);
}
//------------------------------------------------------------------------------
