#ifndef GUARD_GATETOBINARY_HH
#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_FILE

/*!
 *	\file GateToBinary.hh
 *	\brief Store binary data
 *	\author Didier Benoit <benoit@imnc.in2p3.fr>
 *	\date May 2010, IMNC/CNRS, Orsay
 *	\version 1.0
 *
 *	Storing the output data from a system in a binary file
 *
 *	\section LICENCE
 *
 *	Copyright (C): OpenGATE Collaboration
 *	This software is distributed under the terms of the GNU Lesser General
 *	Public Licence (LGPL) See LICENSE.md for further details
 */

#include <string>
#include <vector>
#include <cerrno>
#include <cstdlib>

#include "GateVOutputModule.hh"
#include "GateCoincidenceDigi.hh"
#include "GateDigi.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateRunManager.hh"
#include "GateDigitizerMgr.hh"
class GateToBinaryMessenger;

/*!
 *	\class GateToBinary GateToBinary.hh
 *	\brief GateToBinary class
 *
 *	Storing the output data in a binary file. This class inherits
 *	GateVOutputModule
 */
class GateToBinary : public GateVOutputModule
{
public:
  /*!
   *	\brief Constructor
   *
   *	Constructor of the GateToBinary class
   *
   */
  GateToBinary( G4String const& name, GateOutputMgr* outputMgr,
		DigiMode digiMode );

  /*!
   *	\brief Destructor
   *
   *	Destructor of the GateToBinary class
   *
   */
  virtual ~GateToBinary();

  /*!
   *	\fn virtual void RecordBeginOfAcquisition()
   *	\brief Begin of the acquisition and open the binary file(s)
   */
  virtual void RecordBeginOfAcquisition();

  /*!
   *	\fn virtual void RecordEndOfAcquisition()
   *	\brief End of the acquisition and close the binary file(s)
   */
  virtual void RecordEndOfAcquisition();

  /*!
   *	\fn virtual void RecordBeginOfRun( G4Run const* )
   *	\brief Begin of the run
   */
  virtual void RecordBeginOfRun( G4Run const* );

  /*!
   *	\fn virtual void RecordEndOfRun( G4Run const* )
   *	\brief End of the run and saves the run-specific info in the run file
   */
  virtual void RecordEndOfRun( G4Run const* );

  /*!
   *	\fn virtual void RecordBeginOfEvent( G4Event const* )
   *	\brief Begin of the event
   */
  virtual void RecordBeginOfEvent( G4Event const* );

  /*!
   *	\fn virtual void RecordEndOfEvent( G4Event const* )
   *	\brief End of the event and saves the hits in the binary files, and
   *	calls RecordDigitizer
   *	\param event pointer on the events
   */
  virtual void RecordEndOfEvent( G4Event const* event );

  /*!
   *	\fn virtual void RecordStepWithVolume( GateVVolume const*, G4Step const* )
   *	\brief Record the step with the volume
   */
  virtual void RecordStepWithVolume( GateVVolume const*, G4Step const* );

  /*!
   *	\fn virtual void RecordVoxels( GateVGeometryVoxelStore* voxelStore )
   *	\brief save the geometry voxel information
   */
  virtual void RecordVoxels( GateVGeometryVoxelStore* voxelStore );

  /*!
   *	\fn virtual void RecordDigitizer( G4Event const* )
   *	\brief Called by RecordEndOfEvent to store the Digis in the ASCII
   *	files
   */
  virtual void RecordDigitizer( G4Event const* );

  /*!
   *	\fn inline virtual G4String const& GiveNameOfFile()
   *	\brief return the name of the output file
   *	\return the name of the input file
   */
  inline virtual G4String const& GiveNameOfFile() { return m_fileName; }

  /*!
   *	\fn inline virtual void SetFileName( G4String const aName )
   *	\param aName name of the output file
   *	\brief set the name of the output file
   */
  inline virtual void SetFileName( G4String const aName )
  { m_fileName = aName; };

  /*!
   *	\fn inline virtual G4bool GetOutFileRunsFlag()
   *	\brief flag to decide if it writes or not to the file
   *	\return true/false
   */
  inline virtual G4bool GetOutFileRunsFlag() { return m_outFileRunsFlag; }

  /*!
   *	\fn inline virtual void SetOutFileRunsFlag( G4bool flag )
   *	\brief flag to decide if it writes or not to the file
   *	\param flag true/false
   */
  inline virtual void SetOutFileRunsFlag( G4bool flag )
  { m_outFileRunsFlag = flag; };

  /*!
   *	\fn inline virtual G4bool GetOutFileHitsFlag()
   *	\brief flag to decide if it writes or not to the file
   *	\return true/false
   */
  inline virtual G4bool GetOutFileHitsFlag() { return m_outFileHitsFlag; }

  /*!
   *	\fn inline virtual void SetOutFileHitsFlag( G4bool flag )
   *	\brief flag to decide if it writes or not to the file
   *	\param flag true/false
   */
  inline virtual void SetOutFileHitsFlag( G4bool flag )
  { m_outFileHitsFlag = flag; }

  /*!
   *	\fn inline virtual G4bool GetOutFileVoxelFlag()
   *	\brief flag to decide if it writes or not to the file
   *	\param flag true/false
   */
  inline virtual G4bool GetOutFileVoxelFlag()
  { return m_outFileVoxelFlag; }

  /*!
   *	\fn inline virtual void SetOutFileVoxelFlag( G4bool flag )
   *	\brief flag to decide if it writes or not to the file
   *	\param flag true/false
   */
  inline virtual void SetOutFileVoxelFlag( G4bool flag )
  { m_outFileVoxelFlag = flag; }

  /*!
   *	\fn inline virtual G4int GetRecordFlag()
   *	\brief flag to decide if it writes or not to the file
   *	\param flag true/false
   */
  inline virtual G4int GetRecordFlag() { return m_recordFlag; }

  /*!
   *	\fn inline virtual void SetRecordFlag( G4int flag )
   *	\brief flag to decide if it writes or not to the file
   *	\param flag true/false
   */
  inline virtual void SetRecordFlag( G4int flag ) { m_recordFlag = flag; }

  /*!
   *	\fn virtual void RegisterNewCoincidenceDigiCollection( G4String const& aCollectionName, G4bool outputFlag )
   *	\brief Register a new coincidence digit collection
   *	\param aCollectionName name of the collection
   *	\param outputFlag flag of the output
   */
  virtual void RegisterNewCoincidenceDigiCollection(
                                                    G4String const& aCollectionName, G4bool outputFlag );

  /*!
   *	\fn virtual void RegisterNewSingleDigiCollection( G4String const& aCollectionName, G4bool outputFlag )
   *	\brief Register a new coincidence digit collection
   *	\param aCollectionName name of the collection
   *	\param outputFlag flag of the output
   */
  virtual void RegisterNewSingleDigiCollection(
                                               G4String const& aCollectionName, G4bool outputFlag );

  /*!
   *	\fn inline virtual void SetVerboseLevel( G4int val )
   *	\brief set the verbosity of each module
   *	\param val value of the verbosity
   */
  inline virtual void SetVerboseLevel( G4int val )
  {
    GateVOutputModule::SetVerboseLevel( val );
    for( size_t i = 0; i < m_outputChannelVector.size(); ++i )
      {
        m_outputChannelVector[ i ]->SetVerboseLevel( val );
      }
  }

  /*!
   *	\struct VOutputChannel GateToBinary.hh
   *	\brief VOutputChannel structure
   *
   *	This structure handles the channels. This structure is inherited by
   *	the SingleOutputChannel and CoincidenceOutputChannel structures
   */
  typedef struct VOutputChannel
  {
  public:
    /*!
     *	\brief Constructor
     *
     *	Constructor of the VOutputChannel class
     *
     */
    VOutputChannel( G4String const& aCollectionName, G4bool outputFlag )
      : nVerboseLevel( 0 ), m_outputFlag( outputFlag ),
        m_fileBaseName( G4String( "" ) ),
        m_collectionName( aCollectionName ), m_fileCounter( 0 ),
        m_collectionID( -1 ),
		m_signlesCommands(0)
    {}

    /*!
     *	\brief Destructor
     *
     *	Destructor of the VOutputChannel class
     *
     */
    virtual ~VOutputChannel() {}

    /*!
     *	\fn virtual void RecordDigitizer()
     *	\brief Virtual pure method. Record the digitizer
     */
    virtual void RecordDigitizer() = 0;

    /*!
     *	\fn virtual void OpenFile( G4String const& aFileBaseName )
     *	\param aFileBaseName name of base of the output
     *	\brief open each output file of the channel
     */
    virtual void OpenFile( G4String const& aFileBaseName );

    /*!
     *	\fn virtual void CloseFile()
     *	\brief close each output file of the channel
     */
    virtual void CloseFile();

    /*!
     *	\fn virtual G4bool ExceedsSize()
     *	\brief check if the limit size is exceeded
     */
    virtual G4bool ExceedsSize();

    /*!
     *	\fn inline virtual void SetOutputFlag( G4bool flag )
     *	\param flag true/false
     *	\brief set the output flag
     */
    inline virtual void SetOutputFlag( G4bool flag )
    { m_outputFlag = flag; };

    /*!
     *	\fn inline virtual void SetVerboseLevel( G4int val )
     *	\param val value of verbose
     *	\brief set the value of verbose
     */
    inline virtual void SetVerboseLevel( G4int val )
    { nVerboseLevel = val; };

    /*!
     *	\fn inline static void SetOutputFileSizeLimit( G4int limit )
     *	\param limit limit of the output file(s)
     *	\brief set the size of the output file (in byte)
     */
    inline static void SetOutputFileSizeLimit( G4int limit )
    { m_outputFileSizeLimit = limit; };

    inline void AddSinglesCommand() { m_signlesCommands++; };


  public:
    G4int nVerboseLevel; /*!< Level of verbose */
    G4bool m_outputFlag; /*!< Flag of output */
    G4String m_fileBaseName; /*!< Name of the base of the file */
    G4String m_collectionName; /*!< Name of the collection */
    G4int m_fileCounter; /*!< Count of the file */
    G4int	m_collectionID; /*!< Collection ID */
    std::ofstream m_outputFile; /*!< Output file */
    static G4int m_outputFileSizeLimit; /*!< Output file size limit */
    G4int m_signlesCommands;

  } VOutputChannel;

  /*!
   *	\struct SingleOutputChannel GateToBinary.hh
   *	\brief SingleOutputChannel structure
   *
   *	This structure handles the single channel. This structure inherits
   *	the VOutputChannel structure
   */
  typedef struct SingleOutputChannel : public VOutputChannel
  {
  public:
    /*!
     *	\brief Constructor
     *
     *	Constructor of the SingleOutputChannel class
     *
     */
	  SingleOutputChannel( G4String const& aCollectionName,
	                          G4bool outputFlag )
	       : GateToBinary::VOutputChannel( aCollectionName, outputFlag )
	     {}

    /*!
     *	\brief Destructor
     *
     *	Destructor of the SingleOutputChannel class
     *
     */
    virtual ~SingleOutputChannel() {}

    /*!
     *	\fn virtual void RecordDigitizer()
     *	\brief Record the digitizer
     */
    virtual void RecordDigitizer();

    void OpenFile(G4String const& aFileBaseName )
    {
      // if it's not the first file with the same name, add a suffix like _001
      //to the file name, before .dat
      if( ( m_fileCounter > 0 ) && ( m_fileBaseName != aFileBaseName ) )
        {
          m_fileCounter = 0;
        }

      G4String fileCounterSuffix( "" );
      if( m_fileCounter > 0 )
        {
          std::ostringstream oss;
          oss << std::setfill( '0' ) << std::setw( 3 ) << m_fileCounter;
          fileCounterSuffix = G4String("_") + oss.str();
        }

      //OK GND 2022 multiSD backward compatibility
      GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
      G4String fileName;
      if( digitizerMgr->m_SDlist.size()==1 )
      {
    	  if(m_signlesCommands==0)
    	  {
    	  std::string tmp_str = m_collectionName.substr(0, m_collectionName.find("_"));
    	  fileName = aFileBaseName + tmp_str + fileCounterSuffix + ".bin";
    	  }
    	  else
    		  fileName =  aFileBaseName + m_collectionName + fileCounterSuffix + ".bin";

    }
      else
    	  fileName =  aFileBaseName + m_collectionName + fileCounterSuffix + ".bin";
      if( m_outputFlag )
        {
          m_outputFile.open( fileName.c_str(), std::ios::out |
                             std::ios::binary );
        }
      m_fileBaseName = aFileBaseName;
      ++m_fileCounter;
    }



  } SingleOutputChannel;

  /*!
   *	\struct CoincidenceOutputChannel GateToBinary.hh
   *	\brief CoincidenceOutputChannel structure
   *
   *	This structure handles coincidence channel. This structure inherits
   *	the VOutputChannel structure
   */
  typedef struct CoincidenceOutputChannel : public VOutputChannel
  {
  public:
    /*!
     *	\brief Constructor
     *
     *	Constructor of the CoincidenceOutputChannel class
     *
     */
    CoincidenceOutputChannel( G4String const& aCollectionName,
                              G4bool outputFlag)
      : GateToBinary::VOutputChannel( aCollectionName, outputFlag )
    {}

    /*!
     *	\brief Destructor
     *
     *	Destructor of the CoincidenceOutputChannel class
     *
     */
    virtual ~CoincidenceOutputChannel() {}

    /*!
     *	\fn virtual void RecordDigitizer()
     *	\brief Record the digitizer
     */
    virtual void RecordDigitizer();
  } CoincidenceOutputChannel;

protected:
  GateToBinaryMessenger* m_binaryMessenger; /*!< pointer on the binary messenger */

  G4String m_fileName; /*!< Name of the output file */
  G4bool m_outFileHitsFlag; /*!< Flag for the hits outfile */
  G4bool m_outFileVoxelFlag; /*!< Flag for the voxel outfile */
  G4bool m_outFileRunsFlag; /*!< Flag for the run outfile */
  G4int m_recordFlag; /*!< Record Flag */
  std::vector< VOutputChannel* > m_outputChannelVector; /*!< Vector of output channel */

  std::ofstream m_outFileRun; /*!< outfile for run */
  //std::ofstream m_outFileHits; /*!< outfile for hits */
  //OK GND 2022
  std::vector<std::ofstream> m_outFilesHits; /*!< outfile for hits */
  G4int   m_nSD; // number of sensitive detectors
  //OK GND 2002

private:
  static G4String FixedWidthZeroPaddedString(const G4String & full, size_t length);
};

#endif
#endif
