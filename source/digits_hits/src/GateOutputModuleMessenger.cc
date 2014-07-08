/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateOutputModuleMessenger.hh"
#include "GateVOutputModule.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateOutputModuleMessenger::GateOutputModuleMessenger(GateVOutputModule* outputModule)
  : GateMessenger(outputModule->GetOutputMgr()->GetName()+ G4String("/") + outputModule->GetName()),
    m_outputModule(outputModule)
{

  G4String cmdName;

  cmdName = GetDirectoryName()+"describe";
  DescribeCmd = new G4UIcmdWithoutParameter(cmdName,this);
  DescribeCmd->SetGuidance("List of the outputModule properties");

  cmdName = GetDirectoryName()+"verbose";
  VerboseCmd = new G4UIcmdWithAnInteger(cmdName,this);
  VerboseCmd->SetGuidance("Set GATE output module verbose level");
  VerboseCmd->SetGuidance("1. Integer verbose level");
  VerboseCmd->SetParameterName("verbose",false);
  VerboseCmd->SetRange("verbose>=0");

  cmdName = GetDirectoryName()+"enable";
  EnableCmd = new G4UIcmdWithoutParameter(cmdName,this);
  G4String guidance = G4String("Enables '") + GetDirectoryName() + "'.";
  EnableCmd->SetGuidance(guidance.c_str());

  cmdName = GetDirectoryName()+"disable";
  DisableCmd = new G4UIcmdWithoutParameter(cmdName,this);
  guidance = G4String("Disables '") + GetDirectoryName() + "'.";
  DisableCmd->SetGuidance(guidance.c_str());
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateOutputModuleMessenger::~GateOutputModuleMessenger()
{
  delete DescribeCmd;
  delete VerboseCmd;
  delete EnableCmd;
  delete DisableCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateOutputModuleMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command == VerboseCmd ) {
    m_outputModule->SetVerboseLevel(VerboseCmd->GetNewIntValue(newValue));
  } else if( command == DescribeCmd ) {
    m_outputModule->Describe();
  } else if ( command==EnableCmd ) {
    m_outputModule->Enable(true);
  } else if ( command==DisableCmd ) {
    m_outputModule->Enable(false);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
