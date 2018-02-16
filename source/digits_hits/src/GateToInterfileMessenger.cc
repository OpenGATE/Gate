/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToInterfileMessenger.hh"
#include "GateToInterfile.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

GateToInterfileMessenger::GateToInterfileMessenger(GateToInterfile* gateToInterfile)
  : GateOutputModuleMessenger(gateToInterfile)
  , m_gateToInterfile(gateToInterfile)
{
  G4String cmdName;

/*
  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output Interfile file");
  SetFileNameCmd->SetParameterName("Name",false);
*/
}

GateToInterfileMessenger::~GateToInterfileMessenger()
{
//  delete SetFileNameCmd;
}

void GateToInterfileMessenger::SetNewValue(G4UIcommand* command,G4String /*newValue*/)
{
/*
  if (command == SetFileNameCmd)
    {  m_gateToInterfile->SetFileName(newValue); }
*/
  // All mother macro commands are overloaded to do nothing
  if( command == GetVerboseCmd() ) {
    G4cout << "GateToInterfile::VerboseCmd: Do nothing\n";
  } else if( command == GetDescribeCmd() ) {
    G4cout << "GateToInterfile::DescribeCmd: Do nothing\n";
  } else if ( command == GetEnableCmd() ) {
    G4cout << "GateToInterfile::EnableCmd: Do nothing\n";
  } else if ( command == GetDisableCmd() ) {
    G4cout << "GateToInterfile::DisableCmd: Do nothing\n";
  }
/* No else anymore
  else
    { GateOutputModuleMessenger::SetNewValue(command,newValue);  }
*/
}
