/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToSummaryMessenger.hh"
#include "GateToSummary.hh"

#ifdef G4ANALYSIS_USE_FILE

#include "GateOutputMgr.hh"
#include "GateCoincidenceDigi.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithoutParameter.hh"

//--------------------------------------------------------------------------------
GateToSummaryMessenger::GateToSummaryMessenger(GateToSummary* gateToSummary)
  : GateOutputModuleMessenger(gateToSummary)
  , m_gateToSummary(gateToSummary)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output data file");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName() + "addCollection";
  m_addCollectionCmd = new G4UIcmdWithAString(cmdName, this);
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
GateToSummaryMessenger::~GateToSummaryMessenger()
{
  delete SetFileNameCmd;
  delete m_addCollectionCmd;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummaryMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if (command == SetFileNameCmd) m_gateToSummary->SetFileName(newValue);
  else if (command == m_addCollectionCmd) m_gateToSummary->addCollection(newValue);
  else GateOutputModuleMessenger::SetNewValue(command, newValue);
}
//--------------------------------------------------------------------------------


#endif
