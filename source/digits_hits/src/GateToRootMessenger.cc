/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*  Optical Photons: V. Cuplov -  2012
         - Revision 2012/09/17  /gate/output/root/setRootOpticalFlag functionality added.
           Set the flag for Optical ROOT output.
*/


#include "GateToRootMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateToRoot.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//--------------------------------------------------------------------------
GateToRootMessenger::GateToRootMessenger(GateToRoot* gateToRoot)
  : GateOutputModuleMessenger(gateToRoot)
  , m_gateToRoot(gateToRoot)
{
//  G4cout << " Constructor GateToRootMessenger" << G4endl;
  G4String cmdName;

  cmdName = GetDirectoryName()+"reset";
  ResetCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ResetCmd->SetGuidance("Reset the output");

  cmdName = GetDirectoryName()+"setFileName";
//  G4cout << " cmdName setFileName = " << cmdName << G4endl;
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output ROOT data file");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"setRootHitFlag";
  RootHitCmd = new G4UIcmdWithABool(cmdName,this);
  RootHitCmd->SetGuidance("Set the flag for Hits ROOT output");
  RootHitCmd->SetGuidance("1. true/false");

  cmdName = GetDirectoryName()+"setRootNtupleFlag";
  RootNtupleCmd = new G4UIcmdWithABool(cmdName,this);
  RootNtupleCmd->SetGuidance("Set the flag for Ntuples ROOT output");
  RootNtupleCmd->SetGuidance("1. true/false");

// optical data
  cmdName = GetDirectoryName()+"setRootOpticalFlag";
  RootOpticalCmd = new G4UIcmdWithABool(cmdName,this);
  RootOpticalCmd->SetGuidance("Set the flag for Optical ROOT output");
  RootOpticalCmd->SetGuidance("1. true/false");
// optical data

  cmdName = GetDirectoryName()+"setRootRecordFlag";
  RootRecordCmd = new G4UIcmdWithABool(cmdName,this);
  RootRecordCmd->SetGuidance("Set the flag for Histogram ROOT output");
  RootRecordCmd->SetGuidance("1. true/false");


  cmdName = GetDirectoryName()+"setSaveRndmFlag";
  SaveRndmCmd = new G4UIcmdWithABool(cmdName,this);
  SaveRndmCmd->SetGuidance("Set the flag for change the seed at each Run");
  SaveRndmCmd->SetGuidance("1. true/false");

}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
GateToRootMessenger::~GateToRootMessenger()
{
  delete ResetCmd;

  delete RootHitCmd;
  delete RootNtupleCmd;
  delete RootOpticalCmd;
  delete SetFileNameCmd;
  delete SaveRndmCmd;
  for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i)
    delete OutputChannelCmdList[i];
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRootMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{

  if( command == ResetCmd ) {
    m_gateToRoot->Reset();
  } else if (command == SetFileNameCmd) {
    m_gateToRoot->SetFileName(newValue);
  } else if (command == RootHitCmd) {
    m_gateToRoot->SetRootHitFlag(RootHitCmd->GetNewBoolValue(newValue));
  } else if (command == SaveRndmCmd) {
    m_gateToRoot->SetSaveRndmFlag(SaveRndmCmd->GetNewBoolValue(newValue));
  } else if (command == RootNtupleCmd) {
    m_gateToRoot->SetRootNtupleFlag(RootNtupleCmd->GetNewBoolValue(newValue));
  } else if (command == RootOpticalCmd) {
    m_gateToRoot->SetRootOpticalFlag(RootOpticalCmd->GetNewBoolValue(newValue));
  } else if (command == RootRecordCmd) {
	  m_gateToRoot->SetRecordFlag(RootRecordCmd->GetNewBoolValue(newValue));
	} else if ( IsAnOutputChannelCmd(command) ) {

    ExecuteOutputChannelCmd(command,newValue);
  } else {
    GateOutputModuleMessenger::SetNewValue(command,newValue);
  }

}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRootMessenger::CreateNewOutputChannelCommand(GateToRoot::VOutputChannel* anOutputChannel)
{

  GateMessage("OutputMgr", 5, " GateToRootMessenger::CreateNewOutputChannelCommand -- begin " << G4endl;);

  G4String cmdName;

  m_outputChannelList.push_back(anOutputChannel);

  G4String channelName = anOutputChannel->m_collectionName;
  cmdName = GetDirectoryName()+"setRoot" + channelName + "Flag";

  G4UIcmdWithABool * newCmd = new G4UIcmdWithABool(cmdName,this) ;
  G4String aGuidance = "Set the flag for ROOT output of " + channelName + ".";
  newCmd->SetGuidance(aGuidance.c_str());
  newCmd->SetGuidance("1. true/false");
  OutputChannelCmdList.push_back( newCmd );
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
G4bool GateToRootMessenger::IsAnOutputChannelCmd(G4UIcommand* command)
{
  for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i)
    if ( command == OutputChannelCmdList[i] )
      return true;
  return false;
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateToRootMessenger::ExecuteOutputChannelCmd(G4UIcommand* command, G4String newValue)
{
  for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i){
    if ( command == OutputChannelCmdList[i] ) {
      m_outputChannelList[i]->SetOutputFlag( OutputChannelCmdList[i]->GetNewBoolValue(newValue) );
      break;
    }
    }
}
//--------------------------------------------------------------------------

#endif
