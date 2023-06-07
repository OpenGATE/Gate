/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToSummaryMessenger.hh"
#include "GateToSummary.hh"

#include "GateDigitizerMgr.hh"
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

	GateDigitizerMgr* digitizerMgr=GateDigitizerMgr::GetInstance();

  if (command == SetFileNameCmd) m_gateToSummary->SetFileName(newValue);
  else if (command == m_addCollectionCmd)
  {
	  //OK GND 2022
	  for(size_t i=0;i<digitizerMgr->m_SingleDigitizersList.size();i++)
		  {
		if (newValue==digitizerMgr->m_SingleDigitizersList[i]->GetName()) //save all collections
				  {
					digitizerMgr->m_SingleDigitizersList[i]->m_recordFlag=true;
					m_gateToSummary->addCollection(digitizerMgr->m_SingleDigitizersList[i]->GetOutputName());
				  }
		else if (G4StrUtil::contains(newValue, "_") )
		 { //save only one specific collections

			 m_gateToSummary->addCollection(newValue);
			 GateSinglesDigitizer* digitizer=digitizerMgr->FindSinglesDigitizer(newValue);

			 if(digitizer)
				 digitizer->m_recordFlag=true;
		 }
		  }

	if (newValue=="Coincidences")
		m_gateToSummary->addCollection(newValue);

	//Setting flag in the digitizerMgr
	if (G4StrUtil::contains(newValue, "Singles"))
	{
		digitizerMgr->m_recordSingles=true;
	}
	if (G4StrUtil::contains(newValue, "Coincidences"))
	{

		digitizerMgr->m_recordCoincidences=true;
	}



  }
	  else GateOutputModuleMessenger::SetNewValue(command, newValue);
}
//--------------------------------------------------------------------------------


#endif
