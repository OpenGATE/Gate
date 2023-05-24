/*!
 *	\date May 2010, IMNC/CNRS, Orsay
 *
 *	\section LICENCE
 *
 *	Copyright (C): OpenGATE Collaboration
 *	This software is distributed under the terms of the GNU Lesser General
 *	Public Licence (LGPL) See LICENSE.md for further details
 */

#include "GateToBinary.hh"
#include "GateToBinaryMessenger.hh"
#include "GateDigitizerMgr.hh"

#ifdef G4ANALYSIS_USE_FILE

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"

GateToBinaryMessenger::GateToBinaryMessenger( GateToBinary* gateToBinary )
  :	GateOutputModuleMessenger( gateToBinary ),
	m_gateToBinary( gateToBinary ), m_coincidenceMaskLength( 100 ),
	m_singleMaskLength( 100 )
{
  G4String cmdName;

  cmdName = GetDirectoryName() + "setFileName";
  m_setFileNameCmd = new G4UIcmdWithAString( cmdName, this );
  m_setFileNameCmd->SetGuidance(
                                "Set the name of the output Binary data files" );
  m_setFileNameCmd->SetParameterName( "Name", false );

  cmdName = GetDirectoryName() + "setOutFileHitsFlag";
  m_outFileHitsCmd = new G4UIcmdWithABool( cmdName, this );
  m_outFileHitsCmd->SetGuidance( "Set the flag for Hits Binary output" );
  m_outFileHitsCmd->SetGuidance( "1. true/false" );

  // OK GND 2022
   cmdName = GetDirectoryName()+"setOutFileSinglesFlag";
   m_outFileSinglesCmd = new G4UIcmdWithABool(cmdName,this);
   m_outFileSinglesCmd->SetGuidance("To get error if you use old command");
   m_outFileSinglesCmd->SetGuidance("1. true/false");
   //OK GND 2022


  cmdName = GetDirectoryName() + "setOutFileVoxelFlag";
  m_outFileVoxelCmd = new G4UIcmdWithABool( cmdName, this );
  m_outFileVoxelCmd->SetGuidance(
                                 "Set the flag for the Voxel Material Matrix Binary output" );
  m_outFileVoxelCmd->SetGuidance( "1. true/false" );

  cmdName = GetDirectoryName() + "setCoincidenceMask";
  m_coincidenceMaskCmd = new G4UIcommand( cmdName, this );
  m_coincidenceMaskCmd->SetGuidance(
                                    "Set the mask for the coincidence Binary output" );
  m_coincidenceMaskCmd->SetGuidance( "Sequence of 0 / 1" );

  for( G4int iMask = 0; iMask < m_coincidenceMaskLength; ++iMask )
    {
      G4UIparameter* MaskParam = new G4UIparameter( "mask", 'b', true );
      MaskParam->SetDefaultValue( false );
      m_coincidenceMaskCmd->SetParameter( MaskParam );
    }

  cmdName = GetDirectoryName() + "setSingleMask";
  m_singleMaskCmd = new G4UIcommand( cmdName, this );
  m_singleMaskCmd->SetGuidance(
                               "Set the mask for the single Binary output" );
  m_singleMaskCmd->SetGuidance( "Sequence of 0 / 1" );

  for( G4int iMask = 0; iMask < m_singleMaskLength; ++iMask )
    {
      G4UIparameter* MaskParam = new G4UIparameter( "mask", 'b', true );
      MaskParam->SetDefaultValue( false );
      m_singleMaskCmd->SetParameter( MaskParam );
    }

  cmdName = GetDirectoryName()+"setOutFileSizeLimit";
  m_setOutFileSizeLimitCmd = new G4UIcmdWithAnInteger( cmdName, this );
  m_setOutFileSizeLimitCmd->SetGuidance(
                                        "Set the limit for the size (bytes) of the output binary data files" );
  m_setOutFileSizeLimitCmd->SetParameterName( "size", false );
}

GateToBinaryMessenger::~GateToBinaryMessenger()
{
  delete m_setOutFileSizeLimitCmd;
  delete m_coincidenceMaskCmd;
  delete m_singleMaskCmd;
  delete m_outFileHitsCmd;
  delete m_outFileSinglesCmd;
  delete m_outFileVoxelCmd;
  delete m_setFileNameCmd;
  // for( size_t i = 0; i < m_outputChannelCmd.size(); ++i )
  //   {
  //     delete m_outputChannelCmd[ i ];
  //   }
}

void GateToBinaryMessenger::SetNewValue( G4UIcommand* command,
                                         G4String newValue )
{
  if( command == m_coincidenceMaskCmd )
    {
      std::vector< G4bool > maskVector;
      std::string newValueString = static_cast< std::string >( newValue );

      std::istringstream iss( newValueString );

      G4int mask( 0 );
      maskVector.clear();

      for( G4int iMask = 0; iMask < m_coincidenceMaskLength; ++iMask)
        {
          iss >> mask;
          maskVector.push_back( mask );
        }
      GateCoincidenceDigi::SetCoincidenceASCIIMask( maskVector );
    }
  else if( command == m_singleMaskCmd )
    {
      std::vector< G4bool > maskVector;
      std::string newValueString = static_cast< std::string >( newValue );

      std::istringstream iss( newValueString );

      G4int mask( 0 );
      maskVector.clear();

      for( G4int iMask = 0; iMask < m_singleMaskLength; ++iMask )
        {
          iss >> mask;
          maskVector.push_back( mask );
        }
      GateDigi::SetSingleASCIIMask( maskVector );
    }
  else if( command == m_setOutFileSizeLimitCmd )
    {
      GateToBinary::VOutputChannel::SetOutputFileSizeLimit(
                                                           m_setOutFileSizeLimitCmd->GetNewIntValue( newValue ) );
    }
  else if( command == m_setFileNameCmd )
    {
      m_gateToBinary->SetFileName( newValue );
    }
  else if ( command == m_outFileHitsCmd )
    {
      m_gateToBinary->SetOutFileHitsFlag(
                                         m_outFileHitsCmd->GetNewBoolValue( newValue ) );
    }
  else if (command == m_outFileSinglesCmd) {

    	  //OK GND backward compatibility
    		GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

    		for(size_t j=0; j<digitizerMgr->m_SDlist.size();j++)
    		{
    			for (size_t i = 0; i<m_outputChannelVector.size() ; ++i)
    			 {
    		  		std::string tmp_str = m_outputChannelVector[i]->m_collectionName.substr(0, m_outputChannelVector[i]->m_collectionName.find("_"));
    		  		//Save only main singles digitizer output and not for all DMs
    				 if (m_outputChannelVector[i]->m_collectionName == tmp_str+"_"+digitizerMgr->m_SDlist[j]->GetName() )
    				 {
    					 m_outputChannelVector[i]->SetOutputFlag( m_outFileSinglesCmd->GetNewBoolValue(newValue));
    					 //G4cout<<"Set flag"<< m_outputChannelList[i]->m_outputFlag<<G4endl;
    				 }


    			 GateSinglesDigitizer* digitizer=digitizerMgr->FindSinglesDigitizer(tmp_str+"_"+digitizerMgr->m_SDlist[j]->GetName());
    			 if(digitizer)
    				 digitizer->m_recordFlag=true;
    			 }

    			 digitizerMgr->m_recordSingles=m_outFileSinglesCmd->GetNewBoolValue(newValue);

    		}

    }
  else if ( command == m_outFileVoxelCmd )
    {
      m_gateToBinary->SetOutFileVoxelFlag(
                                          m_outFileVoxelCmd->GetNewBoolValue( newValue ) );
    }
  else if ( IsAnOutputChannelCmd( command ) )
    {
      ExecuteOutputChannelCmd( command, newValue );
    }
  else
    {
      GateOutputModuleMessenger::SetNewValue( command, newValue );
    }
}

void GateToBinaryMessenger::CreateNewOutputChannelCommand(
                                                          GateToBinary::VOutputChannel* anOutputChannel )
{
  G4String cmdName;

  // Add the output channel
  m_outputChannelVector.push_back( anOutputChannel );

  G4String channelName = anOutputChannel->m_collectionName;
  cmdName = GetDirectoryName()+ "setOutFile" + channelName + "Flag";

  G4UIcmdWithABool* newCmd = new G4UIcmdWithABool( cmdName, this );
  G4String aGuidance = "Set the flag for Binary output of " + channelName
    + ".";
  newCmd->SetGuidance( aGuidance.c_str() );
  newCmd->SetGuidance( "1. true/false" );

  // Add the new command
  m_outputChannelCmd.push_back( newCmd );
}

G4bool GateToBinaryMessenger::IsAnOutputChannelCmd( G4UIcommand* command )
{
  for( size_t i = 0; i < m_outputChannelCmd.size(); ++i )
    {
      if( command == m_outputChannelCmd[ i ] )
        {
          return true;
        }
    }
  return false;
}

void GateToBinaryMessenger::ExecuteOutputChannelCmd( G4UIcommand* command,
                                                     G4String newValue)
{
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

  for( size_t i = 0; i < m_outputChannelCmd.size() ; ++i)
    {
      if( command == m_outputChannelCmd[ i ] )
        {
          m_outputChannelVector[i]->SetOutputFlag(
                                                  m_outputChannelCmd[ i ]->GetNewBoolValue( newValue ) );
          GateSinglesDigitizer* digitizer=digitizerMgr->FindSinglesDigitizer(m_outputChannelVector[i]->m_collectionName);
          if(digitizer)
        	  digitizer->m_recordFlag=true;

          //Setting flag in the digitizerMgr
          if (G4StrUtil::contains(m_outputChannelVector[i]->m_collectionName, "Singles"))
          {
        	  m_outputChannelVector[i]->AddSinglesCommand();
        	  digitizerMgr->m_recordSingles=m_outputChannelCmd[i]->GetNewBoolValue(newValue);
          }
          if (G4StrUtil::contains(m_outputChannelVector[i]->m_collectionName, "Coincidences"))
          {
        	  digitizerMgr->m_recordCoincidences=m_outputChannelCmd[i]->GetNewBoolValue(newValue);
          }
          break;
        }
    }
}

#endif
