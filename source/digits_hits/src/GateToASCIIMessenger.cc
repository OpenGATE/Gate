/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToASCIIMessenger.hh"
#include "GateToASCII.hh"

#ifdef G4ANALYSIS_USE_FILE

#include "GateOutputMgr.hh"
#include "GateCoincidenceDigi.hh"
#include "GateDigitizerMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//--------------------------------------------------------------------------------------------------------
GateToASCIIMessenger::GateToASCIIMessenger(GateToASCII* gateToASCII)
  : GateOutputModuleMessenger(gateToASCII)
  , m_gateToASCII(gateToASCII)
{
  G4String cmdName;

  cmdName = GetDirectoryName()+"reset";
  ResetCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ResetCmd->SetGuidance("Reset the output");

  cmdName = GetDirectoryName()+"setFileName";
  SetFileNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetFileNameCmd->SetGuidance("Set the name of the output ASCII data files");
  SetFileNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"setOutFileHitsFlag";
  OutFileHitsCmd = new G4UIcmdWithABool(cmdName,this);
  OutFileHitsCmd->SetGuidance("Set the flag for Hits ASCII output");
  OutFileHitsCmd->SetGuidance("1. true/false");

  // OK GND 2022
  cmdName = GetDirectoryName()+"setOutFileSinglesFlag";
  OutFileSinglesCmd = new G4UIcmdWithABool(cmdName,this);
  OutFileSinglesCmd->SetGuidance("To get error if you use old command");
  OutFileSinglesCmd->SetGuidance("1. true/false");
  //OK GND 2022


  cmdName = GetDirectoryName()+"setOutFileVoxelFlag";
  OutFileVoxelCmd = new G4UIcmdWithABool(cmdName,this);
  OutFileVoxelCmd->SetGuidance("Set the flag for the Voxel Material Matrix ASCII output");
  OutFileVoxelCmd->SetGuidance("1. true/false");

  cmdName = GetDirectoryName()+"setCoincidenceMask";
  CoincidenceMaskCmd = new G4UIcommand(cmdName,this);
  CoincidenceMaskCmd->SetGuidance("Set the mask for the coincidence ASCII output");
  CoincidenceMaskCmd->SetGuidance("Sequence of 0 / 1");

  m_coincidenceMaskLength = 100;
  for (G4int iMask=0; iMask<m_coincidenceMaskLength; iMask++) {
    G4UIparameter* MaskParam = new G4UIparameter("mask",'b',true);
    MaskParam->SetDefaultValue(false);
    CoincidenceMaskCmd->SetParameter(MaskParam);
  }

  cmdName = GetDirectoryName()+"setSingleMask";
  SingleMaskCmd = new G4UIcommand(cmdName,this);
  SingleMaskCmd->SetGuidance("Set the mask for the single ASCII output");
  SingleMaskCmd->SetGuidance("Sequence of 0 / 1");

  m_singleMaskLength = 100;
  for (G4int iMask=0; iMask<m_singleMaskLength; iMask++) {
    G4UIparameter* MaskParam = new G4UIparameter("mask",'b',true);
    MaskParam->SetDefaultValue(false);
    SingleMaskCmd->SetParameter(MaskParam);
  }

  cmdName = GetDirectoryName()+"setOutFileSizeLimit";
  SetOutFileSizeLimitCmd = new G4UIcmdWithAnInteger(cmdName,this);
  SetOutFileSizeLimitCmd->SetGuidance("Set the limit in bytes for the size of the output ASCII data files");
  SetOutFileSizeLimitCmd->SetParameterName("size",false);

}
//--------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------
GateToASCIIMessenger::~GateToASCIIMessenger()
{
  delete SetOutFileSizeLimitCmd;
  delete CoincidenceMaskCmd;
  delete SingleMaskCmd;
  delete ResetCmd;
  delete OutFileHitsCmd;
  delete OutFileSinglesCmd;
  delete OutFileVoxelCmd;
  delete SetFileNameCmd;
  // for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i)
  //   delete OutputChannelCmdList[i];
}
//--------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------
void GateToASCIIMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command == CoincidenceMaskCmd ) {

    std::vector<G4bool> maskVector;
    const char* newValueChar = newValue;
    //LF
    //std::istrstream is((char*)newValueChar);
    std::istringstream is((char*)newValueChar);
    //
    G4int tempIntBool;
    G4int tempBool;
    maskVector.clear();
    for (G4int iMask=0; iMask<m_coincidenceMaskLength; iMask++) {
      is >> tempIntBool; // NB: is >> bool does not work, so we put is to an integer and we copy the integer to the bool
      tempBool = tempIntBool;
      maskVector.push_back(tempBool);
      //      G4cout << "[GateToASCIIMessenger::SetNewValue] iMask: " << iMask << " maskVector[iMask]: " << maskVector[iMask] << Gateendl;
    }
    GateCoincidenceDigi::SetCoincidenceASCIIMask( maskVector );

  } else if (command == SingleMaskCmd) {

    std::vector<G4bool> maskVector;
    const char* newValueChar = newValue;
    //LF
    //std::istrstream is((char*)newValueChar);
    std::istringstream is((char*)newValueChar);
    //
    G4int tempIntBool;
    G4int tempBool;
    maskVector.clear();
    for (G4int iMask=0; iMask<m_singleMaskLength; iMask++) {
      is >> tempIntBool; // NB: is >> bool does not work, so we put is to an integer and we copy the integer to the bool
      tempBool = tempIntBool;
      maskVector.push_back(tempBool);
      //      G4cout << "[GateToASCIIMessenger::SetNewValue] iMask: " << iMask << " maskVector[iMask]: " << maskVector[iMask] << Gateendl;
    }
    GateDigi::SetSingleASCIIMask( maskVector );

  } else if (command == SetOutFileSizeLimitCmd) {
    GateToASCII::VOutputChannel::SetOutputFileSizeLimit( SetOutFileSizeLimitCmd->GetNewIntValue(newValue));
  } else if (command == ResetCmd) {
    m_gateToASCII->Reset();
  } else if (command == SetFileNameCmd) {
    m_gateToASCII->SetFileName(newValue);
  } else if ( command == OutFileHitsCmd ) {
    m_gateToASCII->SetOutFileHitsFlag(OutFileHitsCmd->GetNewBoolValue(newValue));
  }
  else if (command == OutFileSinglesCmd) {

  	  //OK GND backward compatibility
  		GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

  		for(size_t j=0; j<digitizerMgr->m_SDlist.size();j++)
  		{
  			for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i)
  			 {
  		  		std::string tmp_str = m_outputChannelList[i]->m_collectionName.substr(0, m_outputChannelList[i]->m_collectionName.find("_"));
  		  		//Save only main singles digitizer output and not for all DMs
  				 if (m_outputChannelList[i]->m_collectionName == tmp_str+"_"+digitizerMgr->m_SDlist[j]->GetName() )
  				 {
  					 m_outputChannelList[i]->SetOutputFlag( OutFileSinglesCmd->GetNewBoolValue(newValue));
  					 //G4cout<<"Set flag"<< m_outputChannelList[i]->m_outputFlag<<G4endl;
  				 }


  			 GateSinglesDigitizer* digitizer=digitizerMgr->FindSinglesDigitizer(tmp_str+"_"+digitizerMgr->m_SDlist[j]->GetName());
  			 if(digitizer)
  				 digitizer->m_recordFlag=true;
  			 }

  			 digitizerMgr->m_recordSingles=OutFileSinglesCmd->GetNewBoolValue(newValue);


  		}

  }	else if ( command == OutFileVoxelCmd ) {
    m_gateToASCII->SetOutFileVoxelFlag(OutFileVoxelCmd->GetNewBoolValue(newValue));
  } else if ( IsAnOutputChannelCmd(command) ) {
    ExecuteOutputChannelCmd(command,newValue);
  } else {
    GateOutputModuleMessenger::SetNewValue(command,newValue);
  }

}
//--------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------
void GateToASCIIMessenger::CreateNewOutputChannelCommand(GateToASCII::VOutputChannel* anOutputChannel)
{
  G4String cmdName;

  m_outputChannelList.push_back(anOutputChannel);

  G4String channelName = anOutputChannel->m_collectionName;
  cmdName = GetDirectoryName()+"setOutFile" + channelName + "Flag";

  G4UIcmdWithABool * newCmd = new G4UIcmdWithABool(cmdName,this) ;
  G4String aGuidance = "Set the flag for ASCII output of " + channelName + ".";
  newCmd->SetGuidance(aGuidance.c_str());
  newCmd->SetGuidance("1. true/false");
  OutputChannelCmdList.push_back( newCmd );
}
//--------------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------------
G4bool GateToASCIIMessenger::IsAnOutputChannelCmd(G4UIcommand* command)
{
  for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i)
    if ( command == OutputChannelCmdList[i] )
      return true;
  return false;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateToASCIIMessenger::ExecuteOutputChannelCmd(G4UIcommand* command,G4String newValue)
{
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

  for (size_t i = 0; i<OutputChannelCmdList.size() ; ++i)
    if ( command == OutputChannelCmdList[i] ) {
      m_outputChannelList[i]->SetOutputFlag( OutputChannelCmdList[i]->GetNewBoolValue(newValue) );


      GateSinglesDigitizer* digitizer=digitizerMgr->FindSinglesDigitizer(m_outputChannelList[i]->m_collectionName);
      if(digitizer)
    	  digitizer->m_recordFlag=true;

      //Setting flag in the digitizerMgr
      if (G4StrUtil::contains(m_outputChannelList[i]->m_collectionName, "Singles"))
      {
    	  m_outputChannelList[i]->AddSinglesCommand();
    	  digitizerMgr->m_recordSingles=OutputChannelCmdList[i]->GetNewBoolValue(newValue);
      }


      if (G4StrUtil::contains(m_outputChannelList[i]->m_collectionName, "Coincidences"))
	  {
    	  digitizerMgr->m_recordCoincidences=OutputChannelCmdList[i]->GetNewBoolValue(newValue);
	  }
      break;
    }
}
//--------------------------------------------------------------------------------------------------------

#endif
