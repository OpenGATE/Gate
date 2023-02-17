#ifndef GUARD_GATETOBINARYMESSENGER_HH
#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_FILE

/*!
 *	\file GateToBinaryMessenger.hh
 *	\brief Messenger of GateToBinary class
 *	\author Didier Benoit <benoit@imnc.in2p3.fr>
 *	\date May 2010, IMNC/CNRS, Orsay
 *	\version 1.0
 *
 *	Messenger allowing to set the binary output
 *
 *	\section LICENCE
 *
 *	Copyright (C): OpenGATE Collaboration
 *	This software is distributed under the terms of the GNU Lesser General
 *	Public Licence (LGPL) See LICENSE.md for further details
 */

#include <vector>

#include "GateOutputModuleMessenger.hh"

class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;

class GateToBinary;

/*!
 *	\class GateToBinaryMessenger GateToBinaryMessenger.hh
 *	\brief GateToBinaryMessenger class
 *
 *	Setting the messenger of the output binary data. This class inherits
 *	GateOutputModuleMessenger
 */
class GateToBinaryMessenger : public GateOutputModuleMessenger
{
public:
	/*!
	 *	\brief Constructor
	 *
	 *	Constructor of the GateToBinaryMessenger class
	 *
	 */
	GateToBinaryMessenger( GateToBinary* gateToBinary );

	/*!
	 *	\brief Destructor
	 *
	 *	Destructor of the GateToBinaryMessenger class
	 *
	 */
	~GateToBinaryMessenger();

	/*!
	 *	\fn void SetNewValue( G4UIcommand* command, G4String newValue )
	 *	\brief Set the new value in the messenger
	 *	\param command an ASCII command
	 *	\param newValue
	 */
	void SetNewValue( G4UIcommand* command, G4String newValue );

	/*!
	 *	\fn void CreateNewOutputChannelCommand( GateToBinary::VOutputChannel* anOutputChannel )
	 *	\brief Create a new output channel
	 *	\param anOutputChannel an output channel
	 */
	void CreateNewOutputChannelCommand( GateToBinary::VOutputChannel*
		anOutputChannel );

	/*!
	 *	\fn bool IsAnOutputChannelCmd( G4UIcommand* command )
	 *	\brief Check if the output is an output channel
	 *	\param command an ASCII command
	 *	\return a boolean (true/false) if the output is an output channel or not
	 */
	G4bool IsAnOutputChannelCmd( G4UIcommand* command );

	/*!
	 *	\fn void ExecuteOutputChannelCmd( G4UIcommand* command, G4String newValue)
	 *	\brief Execute the output channel
	 *	\param command an ASCII command
	 *	\param newValue
	 */
	void ExecuteOutputChannelCmd( G4UIcommand* command,
		G4String newValue );

protected:
	GateToBinary* m_gateToBinary; /*!< pointer on the GateToBinary class */

	G4int m_coincidenceMaskLength; /*!< Length of the coincidence mask */
	G4int m_singleMaskLength; /*!< Length of the single mask */

	G4UIcmdWithAString* m_setFileNameCmd; /*!< Command to set the name of file */
	G4UIcmdWithABool* m_outFileHitsCmd; /*!< Command for the hit output */
    G4UIcmdWithABool*  m_outFileSinglesCmd;
	G4UIcmdWithABool* m_outFileVoxelCmd; /*!< Command for the voxel output */
	G4UIcommand* m_coincidenceMaskCmd; /*!< Command for the coincidence mask */
	G4UIcommand* m_singleMaskCmd; /*!< Command for the single mask */
	G4UIcmdWithAnInteger* m_setOutFileSizeLimitCmd; /*!< Limit of the binary output file (in byte) */
	std::vector< G4UIcmdWithABool* > m_outputChannelCmd; /*!< Command for the output */

	std::vector< GateToBinary::VOutputChannel* >  m_outputChannelVector; /*!< vector of output channel */
};

#endif
#endif
