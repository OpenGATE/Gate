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

#ifdef G4ANALYSIS_USE_FILE

#include <limits>
#include <fcntl.h>
#include <unistd.h>

#include "GateToBinaryMessenger.hh"
#include "GateOutputMgr.hh"
#include "GateVGeometryVoxelStore.hh"
#include "G4DigiManager.hh"
#include "GateDigitizerMgr.hh"

// 0x79000000 equivalent to 2,030,043,136 bytes
#define LIMIT_SIZE 0x79000000

G4int GateToBinary::VOutputChannel::m_outputFileSizeLimit = LIMIT_SIZE;

GateToBinary::GateToBinary( G4String const& name, GateOutputMgr* outputMgr,
                            DigiMode digiMode )
  : GateVOutputModule( name, outputMgr, digiMode ),
    m_fileName( " " ),
    m_outFileHitsFlag( digiMode == kruntimeMode ),
    m_outFileVoxelFlag( true ),
    m_outFileRunsFlag( digiMode == kruntimeMode ),
    m_recordFlag( 0 )
{
  // Instanciating the messenger
  m_binaryMessenger = new GateToBinaryMessenger( this );

  // Intializing the mask of the singles and the coincidences to 1
  GateCoincidenceDigi::SetCoincidenceASCIIMask( 1 );
  GateDigi::SetSingleASCIIMask( 1 );
}

GateToBinary::~GateToBinary()
{
  // Deleting the messenger
  delete m_binaryMessenger;
}

void GateToBinary::RecordBeginOfAcquisition()
{
	//OK GND 2022
	 for (size_t i = 0; i < m_outputChannelVector.size(); ++i)
	    {
		 m_outputChannelVector[i]->m_collectionID=-1 ;
	    }



  if( nVerboseLevel > 2 )
    {
      std::cout << "GateToBinary::RecordBeginOfAcquisition\n";
    }

  if( nVerboseLevel > 0 )
    {
      std::cout << "Opening the binary output files...\n";
    }

  if( m_outFileRunsFlag )
    {
      m_outFileRun.open( ( m_fileName + "Run.bin" ).c_str(),
                         std::ios::out | std::ios::binary );
    }

  if( m_outFileHitsFlag )
    {
	  //OK GND 2022
	  	  GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

	  	  m_nSD=digitizerMgr->m_SDlist.size();
	  	  for (G4int i=0; i<m_nSD ;i++)
	  	  {
	  		  std::ofstream outFileHits;

	  		  if (digitizerMgr->m_SDlist.size() ==1 ) // keep the old name "Hits" if there is only one collection
	  			  outFileHits.open((m_fileName+"Hits.bin").c_str(), std::ios::out | std::ios::binary);
	  		  else
	  			  outFileHits.open((m_fileName+"Hits_"+ digitizerMgr->m_SDlist[i]->GetName()+".bin").c_str(), std::ios::out | std::ios::binary);

	  		  m_outFilesHits.push_back(std::move(outFileHits));
	  	  }

    }

  for( size_t i = 0; i < m_outputChannelVector.size(); ++i )
    {
      m_outputChannelVector[ i ]->OpenFile( m_fileName );
    }

  if( nVerboseLevel > 0 )
    {
      std::cout << " ... Binary output files opened\n";
    }
}

void GateToBinary::RecordEndOfAcquisition()
{
  if( nVerboseLevel > 2 )
    {
      G4cout << "GateToBinary::RecordEndOfAcquisition\n";
    }

  // Close the file with the hits information
  if( m_outFileRunsFlag )
    {
      m_outFileRun.close();
    }

  if( m_outFileHitsFlag )
    {
	  //OK GND 2022
	  for (G4int i=0; i< m_nSD;i++)
	  {
		  m_outFilesHits[i].close();
	  }
    }

  for( size_t i = 0; i < m_outputChannelVector.size(); ++i )
    {
      m_outputChannelVector[ i ]->CloseFile();
    }
}

void GateToBinary::RecordBeginOfRun( G4Run const* )
{
  if( nVerboseLevel > 2 )
    {
      std::cout << "GateToBinary::RecordBeginOfRun\n";
    }
}

void GateToBinary::RecordEndOfRun( G4Run const* )
{
  if( nVerboseLevel > 2 )
    {
      std::cout << "GateToBinary::RecordEndOfRun\n";
    }

  if( m_outFileRunsFlag )
    {
      G4int nEvent =
        ( ( (GatePrimaryGeneratorAction*)GateRunManager::GetRunManager()->
            GetUserPrimaryGeneratorAction())->GetEventNumber() );

      if( nVerboseLevel > 0 )
        {
          G4cout << "GateToBinary::RecordEndOfRun: Events in the past run: "
                 << nEvent << Gateendl;
        }

      m_outFileRun.write( reinterpret_cast< char* >( &nEvent ),
                          sizeof( G4int ) );
    }
}

void GateToBinary::RecordBeginOfEvent( G4Event const* )
{
  if( nVerboseLevel > 2 )
    {
      std::cout << "GateToBinary::RecordBeginOfEvent\n";
    }
}

void GateToBinary::RecordEndOfEvent( G4Event const* event )
{
  if( nVerboseLevel > 2 )
    {
      G4cout << "GateToBinary::RecordEndOfEvent\n";
    }

  if( m_outFileHitsFlag )
    {

	  //OK GND 2022
	  std::vector<GateHitsCollection*> CHC_vector = GetOutputMgr()->GetHitCollections();

	 for (size_t i=0; i<CHC_vector.size();i++ )//HC_vector.size()
		{
		 GateHitsCollection* CHC = CHC_vector[i];

      G4int NbHits( 0 );

      if( CHC )
        {
          // Hits loop
          NbHits = CHC->entries();
          for( G4int iHit = 0; iHit < NbHits; ++iHit )
            {
              G4String processName = (*CHC)[ iHit ]->GetProcess();
              G4int PDGEncoding = (*CHC)[ iHit ]->GetPDGEncoding();
              if( nVerboseLevel > 2 )
                {
                  std::cout << "GateToBinary::RecordEndOfEvent : "
                            << "HitsCollection: processName : <"
                            << processName << ">    Particles PDG code : " << PDGEncoding
                            << Gateendl;
                }
              if( (*CHC)[iHit]->GoodForAnalysis() )
                {
                  if( m_outFileHitsFlag )
                    {
                      G4int runID = (*CHC)[ iHit ]->GetRunID();
                      G4int eventID = (*CHC)[ iHit ]->GetEventID();
                      G4int primaryID = (*CHC)[ iHit ]->GetPrimaryID();
                      G4int sourceID = (*CHC)[ iHit ]->GetSourceID();

                      // Element is the number of level
                      size_t const element = 6;
                      G4int volumeID[ element ] = { 0, 0, 0, 0, 0, 0 };

                      // For each level of volume
                      for( size_t lvl = 0;
                           lvl < ( (*CHC)[ iHit ]->GetOutputVolumeID() ).size(); ++lvl )
                        {
                          *( volumeID + lvl ) =
                            (*CHC)[ iHit ]->GetOutputVolumeID()[ lvl ];
                        }

                      G4double timeID = (*CHC)[ iHit ]->GetTime()/s;
                      G4double eDepID = (*CHC)[ iHit ]->GetEdep()/MeV;
                      G4double stepLengthID = (*CHC)[ iHit ]->GetStepLength()/mm;
                      G4double posX = ( (*CHC)[ iHit ]->GetGlobalPos() ).x()/mm;
                      G4double posY = ( (*CHC)[ iHit ]->GetGlobalPos() ).y()/mm;
                      G4double posZ = ( (*CHC)[ iHit ]->GetGlobalPos() ).z()/mm;

                      G4int trackID = (*CHC)[ iHit ]->GetTrackID();
                      G4int parentID = (*CHC)[ iHit ]->GetParentID();
                      G4int photonID = (*CHC)[ iHit ]->GetPhotonID();
                      G4int phCompton = (*CHC)[ iHit ]->GetNPhantomCompton();
                      G4int phRayleigh = (*CHC)[ iHit ]->GetNPhantomRayleigh();

                      G4String compVolName = (*CHC)[ iHit ]->GetComptonVolumeName();
                      G4String rayVolName = (*CHC)[ iHit ]->GetRayleighVolumeName();

                      // Writing data
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &runID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &eventID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &primaryID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &sourceID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write(
                                          reinterpret_cast< char* >( &volumeID[ 0 ] ),
                                          ( (*CHC)[ iHit ]->GetOutputVolumeID() ).size() * sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &timeID ),
                                           sizeof( G4double ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &eDepID ),
                                           sizeof( G4double ) );
                      m_outFilesHits[i].write(
                                          reinterpret_cast< char* >( &stepLengthID ),
                                          sizeof( G4double ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &posX ),
                                           sizeof( G4double ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &posY ),
                                           sizeof( G4double ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &posZ ),
                                           sizeof( G4double ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &PDGEncoding ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &trackID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &parentID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &photonID ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &phCompton ),
                                           sizeof( G4int ) );
                      m_outFilesHits[i].write( reinterpret_cast< char* >( &phRayleigh ),
                                           sizeof( G4int ) );

                      // Previous versions of GATE unintentionally wrote the
                      // structure of G4String (which is std::string) to disk
                      // rather than the string itself.  This was 8 bytes on
                      // most platforms, and referenced as 8 bytes in the
                      // documentaiton.  For this reason we limit the strings
                      // to 8 bytes, or 7 characters with a null terminator.
                      const size_t strFieldWidth = 8;
                      const size_t strMaxLen = strFieldWidth - 1;
                      G4String processNameTrunc = FixedWidthZeroPaddedString(
                                                                             processName, strMaxLen);
                      G4String compVolNameTrunc = FixedWidthZeroPaddedString(
                                                                             compVolName, strMaxLen);
                      G4String rayVolNameTrunc = FixedWidthZeroPaddedString(
                                                                            rayVolName, strMaxLen);
                      m_outFilesHits[i].write( processNameTrunc.c_str(),
                                           strFieldWidth);
                      m_outFilesHits[i].write( compVolNameTrunc.c_str(),
                                           strFieldWidth);
                      m_outFilesHits[i].write( rayVolNameTrunc.c_str(),
                                           strFieldWidth);
                    }
                }// good for analysis
            }//loop over hits
        }//if HC is OK
      else
        {
          if( nVerboseLevel > 0 )
            {
              std::cout <<
                "GateToBinary::RecordHits : GateHitCollection not found"
                        << Gateendl;
            }
        }
	  }// loop over HitCollections
    }
  RecordDigitizer( event );
}

void GateToBinary::RecordDigitizer( G4Event const* )
{
  if( nVerboseLevel > 2 )
    {
      G4cout << "GateToBinary::RecordDigitizer\n";
    }

  for( size_t i = 0; i < m_outputChannelVector.size(); ++i )
    {
	  //OK GND 2022
	  if(m_outputChannelVector[i]->m_collectionID<0)
		  m_outputChannelVector[i]->m_collectionID=GetCollectionID(m_outputChannelVector[i]->m_collectionName);

	  m_outputChannelVector[ i ]->RecordDigitizer();
    }
}

void GateToBinary::RecordStepWithVolume( GateVVolume const*,
                                         G4Step const* )
{
  if( nVerboseLevel > 2 )
    {
      G4cout << "GateToBinary::RecordStep\n";
    }
}

void GateToBinary::RecordVoxels( GateVGeometryVoxelStore* voxelStore )
{
	// TODO !!! OK GND 2020 add (or remove) to GND and documentation

  if( nVerboseLevel > 2 )
    {
      std::cout << "[GateToBinary::RecordVoxels]\n";
    }

  if( m_recordFlag > 0 )
    {
      // protect against huge ASCII files in case of nx,ny,nz ~O(100)
      if( !m_outFileVoxelFlag )
        {
          return;
        }

      //!< Output stream for the voxel density map
      std::ofstream  voxelFile;
      // Open the header file
      G4String voxelFileName = "voxels.dat";
      voxelFile.open( voxelFileName.c_str(), std::ios::out |
                      std::ios::trunc | std::ios::binary );
      if( !( voxelFile.is_open() ) )
        {
          G4String msg = "Could not open the voxel file '" + voxelFileName;
          G4Exception( "GateToBinary::RecordVoxels", "RecordVoxels", FatalException, msg );
        }

      // Write the header: number of voxels, voxel dimensions
      G4int nx = voxelStore->GetVoxelNx();
      G4int ny = voxelStore->GetVoxelNy();
      G4int nz = voxelStore->GetVoxelNz();

      G4ThreeVector voxelSize = voxelStore->GetVoxelSize();

      G4double dx = voxelSize.x()/mm;
      G4double dy = voxelSize.y()/mm;
      G4double dz = voxelSize.z()/mm;

      voxelFile.write( reinterpret_cast< char* >( &nx ), sizeof( G4int ) );
      voxelFile.write( reinterpret_cast< char* >( &ny ), sizeof( G4int ) );
      voxelFile.write( reinterpret_cast< char* >( &nz ), sizeof( G4int ) );
      voxelFile.write( reinterpret_cast< char* >( &dx ),
                       sizeof( G4double ) );
      voxelFile.write( reinterpret_cast< char* >( &dy ),
                       sizeof( G4double ) );
      voxelFile.write( reinterpret_cast< char* >( &dz ),
                       sizeof( G4double ) );

      // Write the content of the voxel matrix
      for( G4int iz = 0; iz < nz; ++iz )
        {
          for( G4int iy = 0; iy < ny; ++iy )
            {
              for( G4int ix = 0; ix < nx; ++ix )
                {
                  G4double density = voxelStore->GetVoxelMaterial( ix, iy, iz )->
                    GetDensity()/(gram/cm3);
                  voxelFile.write( reinterpret_cast< char* >( &density ),
                                   sizeof( G4double ) );
                }
            }
        }
      voxelFile.close();
    }
}

void GateToBinary::RegisterNewSingleDigiCollection(
                                                   G4String const& aCollectionName, G4bool outputFlag )
{
  // Creating a new single digit collection
  VOutputChannel* singleOutputChannel = new SingleOutputChannel(
                                                                aCollectionName, outputFlag );
  m_outputChannelVector.push_back( singleOutputChannel );

  m_binaryMessenger->CreateNewOutputChannelCommand( singleOutputChannel );
}


void GateToBinary::RegisterNewCoincidenceDigiCollection(
                                                        G4String const& aCollectionName, G4bool outputFlag )
{
  // Creating a new coincidence digit collection
  VOutputChannel* coincOutputChannel = new CoincidenceOutputChannel(
                                                                    aCollectionName, outputFlag );
  m_outputChannelVector.push_back( coincOutputChannel );

  m_binaryMessenger->CreateNewOutputChannelCommand( coincOutputChannel );
}

void GateToBinary::VOutputChannel::OpenFile(
                                            G4String const& aFileBaseName )
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

  G4String fileName = aFileBaseName + m_collectionName + fileCounterSuffix
    + ".bin";
  if( m_outputFlag )
    {
      m_outputFile.open( fileName.c_str(), std::ios::out |
                         std::ios::binary );
    }
  m_fileBaseName = aFileBaseName;
  ++m_fileCounter;
}


void GateToBinary::VOutputChannel::CloseFile()
{
  if( m_outputFlag )
    {
      m_outputFile.close();
    }
}

G4bool GateToBinary::VOutputChannel::ExceedsSize()
{
  G4int size = m_outputFile.tellp();
  //std::cout << "size: " << size << " B\n";
  return size > m_outputFileSizeLimit;
}

void GateToBinary::CoincidenceOutputChannel::RecordDigitizer()
{
  G4DigiManager* fDM = G4DigiManager::GetDMpointer();
  /*if( m_collectionID < 0 )
    {
      m_collectionID = fDM->GetDigiCollectionID( m_collectionName );
    }
   */


  GateCoincidenceDigiCollection *CDC =
              (GateCoincidenceDigiCollection *) (fDM->GetDigiCollection(m_collectionID));

  if( !CDC )
    {
      if( nVerboseLevel > 0 )
        {
          G4cout
            << "[GateToBinary::CoincidenceOutputChannel::RecordDigitizer]: "
            << "digi collection '" << m_collectionName << "' not found"
            << Gateendl;
        }
    }
  else
    {
      // Digi loop
      if( nVerboseLevel > 0 )
        {
          G4cout
            << "[GateToBinary::CoincidenceOutputChannel::RecordDigitizer]: "
            << "Totals digits: " << CDC->entries() << Gateendl;
        }
      if( m_outputFlag )
        {
          G4int n_digi =  CDC->entries();
          for( G4int iDigi = 0; iDigi < n_digi; ++iDigi )
            {
              if( m_outputFileSizeLimit > 10000 )
                { // to protect against the creation of too many files by mistake
                  if( ExceedsSize() )
                    {
                      CloseFile();
                      OpenFile( m_fileBaseName );
                    }
                }
              // For the 2 pulses
              G4int runID( 0 ), eventID( 0 ), sourceID( 0 );
              G4double posX( 0.0 ), posY( 0.0 ), posZ( 0.0 ), time( 0.0 );
              G4double sourcePosX( 0.0 ), sourcePosY( 0.0 ), sourcePosZ( 0.0 );
              G4double energy( 0.0 ), scannerPosZ( 0.0 ), scannerRotAng( 0.0 );
              G4int nPhantCompt( 0 ), nCrysCompt( 0 );
              G4int nPhantRay( 0 ), nCrysRay( 0 );
              size_t const element = 6;
              G4int volumeID[ element ] = { 0, 0, 0, 0, 0, 0 };

              for( G4int iP = 0; iP < 2; ++iP )
                {
                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 0 ) )
                    {
                      runID = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetRunID();
                      m_outputFile.write( reinterpret_cast< char* >( &runID ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 1 ) )
                    {
                      eventID = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetEventID();
                      m_outputFile.write( reinterpret_cast< char* >( &eventID ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 2 ) )
                    {
                      sourceID = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetSourceID();
                      m_outputFile.write( reinterpret_cast< char* >( &sourceID ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 3 ) )
                    {
                      sourcePosX = ( (*CDC)[ iDigi ]->
                              GetDigi( iP ) )->GetSourcePosition().x()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &sourcePosX ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 4 ) )
                    {
                      sourcePosY = ( (*CDC)[ iDigi ]->
                              GetDigi( iP ) )->GetSourcePosition().y()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &sourcePosY ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 5 ) )
                    {
                      sourcePosZ = ( (*CDC)[ iDigi ]->
                              GetDigi( iP ) )->GetSourcePosition().z()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &sourcePosZ ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 6 ) )
                    {
                      time = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetTime()/s;
                      m_outputFile.write( reinterpret_cast< char* >( &time ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 7 ) )
                    {
                      energy = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetEnergy()/MeV;
                      m_outputFile.write( reinterpret_cast< char* >( &energy ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 8 ) )
                    {
                      posX = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetGlobalPos().x()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &posX ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 9 ) )
                    {
                      posY = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetGlobalPos().y()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &posY ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 10 ) )
                    {
                      posZ = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetGlobalPos().z()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &posZ ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 11 ) )
                    {
                      // For each level of volume
                      for( size_t lvl = 0;
                              lvl < ( ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                                   GetOutputVolumeID() ).size(); ++lvl )
                        {
                          *( volumeID + lvl ) =
                                  ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                            GetOutputVolumeID()[ lvl ];
                        }
                      m_outputFile.write(
                                         reinterpret_cast< char* >( &volumeID[ 0 ] ),
                                         ( ( (*CDC)[ iDigi ]->GetDigi( iP ) )->GetOutputVolumeID() ).size() * sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 12 ) )
                    {
                      nPhantCompt = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                        GetNPhantomCompton();
                      m_outputFile.write( reinterpret_cast< char* >( &nPhantCompt ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 13 ) )
                    {
                      nCrysCompt = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                        GetNCrystalCompton();
                      m_outputFile.write( reinterpret_cast< char* >( &nCrysCompt ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 14 ) )
                    {
                      nPhantRay = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                        GetNPhantomRayleigh();
                      m_outputFile.write( reinterpret_cast< char* >( &nPhantRay ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 15 ) )
                    {
                      nCrysRay = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                        GetNCrystalRayleigh();
                      m_outputFile.write( reinterpret_cast< char* >( &nCrysRay ),
                                          sizeof( G4int ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 16 ) )
                    {
                      scannerPosZ = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                        GetScannerPos().z()/mm;
                      m_outputFile.write( reinterpret_cast< char* >( &scannerPosZ ),
                                          sizeof( G4double ) );
                    }

                  if ( GateCoincidenceDigi::GetCoincidenceASCIIMask( 17 ) )
                    {
                      scannerRotAng = ( (*CDC)[ iDigi ]->GetDigi( iP ) )->
                        GetScannerRotAngle()/deg;
                      m_outputFile.write( reinterpret_cast< char* >( &scannerRotAng ),
                                          sizeof( G4double ) );
                    }
                }
            }
        }
    }
}

void GateToBinary::SingleOutputChannel::RecordDigitizer()
{
  G4DigiManager* fDM = G4DigiManager::GetDMpointer();
  /*if( m_collectionID < 0 )
    {
      m_collectionID = fDM->GetDigiCollectionID( m_collectionName );
    }
    */
  GateDigiCollection const* SDC =
    (GateDigiCollection*)
    ( fDM->GetDigiCollection( m_collectionID ) );

  if( !SDC )
    {
      if( nVerboseLevel > 0 )
        {
          std::cout << "[GateToBinary::SingleOutputChannel::RecordDigitizer]: "
                    << "digi collection '" << m_collectionName << "' not found"
                    << Gateendl;
        }
    }
  else
    {
      // Digi loop
      if( nVerboseLevel > 0 )
        {
          std::cout << "[GateToBinary::SingleOutputChannel::RecordDigitizer]:"
            "Totals digits: " << SDC->entries() << Gateendl;
        }
      if( m_outputFlag )
        {
          G4int n_digi =  SDC->entries();
          for( G4int iDigi = 0; iDigi < n_digi; ++iDigi)
            {
              if( m_outputFileSizeLimit > 10000 )
                { // to protect against the creation of too many files by mistake
                  if( ExceedsSize() )
                    {
                      CloseFile();
                      OpenFile( m_fileBaseName );
                    }
                }

              G4int runID( 0 ), eventID( 0 ), sourceID( 0 );
              G4double sourcePosX( 0.0 ), sourcePosY( 0.0 ), sourcePosZ( 0.0 );
              G4double posX( 0.0 ), posY( 0.0 ), posZ( 0.0 ), time( 0.0 );
              G4double energy( 0.0 );
              G4int nPhantCompt( 0 ), nCrysCompt( 0 );
              G4int nPhantRay( 0 ), nCrysRay( 0 );
              G4String compVolName( "" ), rayVolName( "" );

              size_t const element = 6;
              G4int volumeID[ element ] = { 0, 0, 0, 0, 0, 0 };

              if ( GateDigi::GetSingleASCIIMask( 0 ) )
                {
                  runID = (*SDC)[ iDigi ]->GetRunID();
                  m_outputFile.write( reinterpret_cast< char* >( &runID ),
                                      sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 1 ) )
                {
                  eventID = (*SDC)[ iDigi ]->GetEventID();
                  m_outputFile.write( reinterpret_cast< char* >( &eventID ),
                                      sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 2 ) )
                {
                  sourceID = (*SDC)[ iDigi ]->GetSourceID();
                  m_outputFile.write( reinterpret_cast< char* >( &sourceID ),
                                      sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 3 ) )
                {
                  sourcePosX = (*SDC)[ iDigi ]->GetSourcePosition().x()/mm;
                  m_outputFile.write( reinterpret_cast< char* >( &sourcePosX ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 4 ) )
                {
                  sourcePosY = (*SDC)[ iDigi ]->GetSourcePosition().y()/mm;
                  m_outputFile.write( reinterpret_cast< char* >( &sourcePosY ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 5 ) )
                {
                  sourcePosZ = (*SDC)[ iDigi ]->GetSourcePosition().z()/mm;
                  m_outputFile.write( reinterpret_cast< char* >( &sourcePosZ ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 6 ) )
                {
                  // For each level of volume
                  for( size_t lvl = 0;
                       lvl < ( (*SDC)[ iDigi ]->GetOutputVolumeID() ).size();
                       ++lvl )
                    {
                      *( volumeID + lvl ) = (*SDC)[ iDigi ]->
                        GetOutputVolumeID()[ lvl ];
                    }
                  m_outputFile.write(
                                     reinterpret_cast< char* >( &volumeID[ 0 ] ),
                                     ( (*SDC)[ iDigi ]->GetOutputVolumeID() ).size() * sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 7 ) )
                {
                  time = (*SDC)[ iDigi ]->GetTime()/s;
                  m_outputFile.write( reinterpret_cast< char* >( &time ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 8 ) )
                {
                  energy = (*SDC)[ iDigi ]->GetEnergy()/MeV;
                  m_outputFile.write( reinterpret_cast< char* >( &energy ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 9 ) )
                {
                  posX = (*SDC)[ iDigi ]->GetGlobalPos().x()/mm;
                  m_outputFile.write( reinterpret_cast< char* >( &posX ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 10 ) )
                {
                  posY = (*SDC)[ iDigi ]->GetGlobalPos().y()/mm;
                  m_outputFile.write( reinterpret_cast< char* >( &posY ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 11 ) )
                {
                  posZ = (*SDC)[ iDigi ]->GetGlobalPos().z()/mm;
                  m_outputFile.write( reinterpret_cast< char* >( &posZ ),
                                      sizeof( G4double ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 12 ) )
                {
                  nPhantCompt = (*SDC)[ iDigi ]->GetNPhantomCompton();
                  m_outputFile.write( reinterpret_cast< char* >( &nPhantCompt ),
                                      sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 13 ) )
                {
                  nCrysCompt = (*SDC)[ iDigi ]->GetNCrystalCompton();
                  m_outputFile.write( reinterpret_cast< char* >( &nCrysCompt ),
                                      sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 14 ) )
                {
                  nPhantRay = (*SDC)[ iDigi ]->GetNPhantomRayleigh();
                  m_outputFile.write( reinterpret_cast< char* >( &nPhantRay ),
                                      sizeof( G4int ) );
                }

              if ( GateDigi::GetSingleASCIIMask( 15 ) )
                {
                  nCrysRay = (*SDC)[ iDigi ]->GetNCrystalRayleigh();
                  m_outputFile.write( reinterpret_cast< char* >( &nCrysRay ),
                                      sizeof( G4int ) );
                }

              // Previous versions of GATE unintentionally wrote the
              // structure of G4String (which is std::string) to disk
              // rather than the string itself.  This was 8 bytes on
              // most platforms, and referenced as 8 bytes in the
              // documentaiton.  For this reason we limit the strings
              // to 8 bytes, or 7 characters with a null terminator.
              const size_t strFieldWidth = 8;
              const size_t strMaxLen = strFieldWidth - 1;

              if ( GateDigi::GetSingleASCIIMask( 16 ) )
                {
                  compVolName = (*SDC)[ iDigi ]->GetComptonVolumeName();
                  G4String compVolNameTrunc = FixedWidthZeroPaddedString(
                                                                         compVolName, strMaxLen);
                  m_outputFile.write( compVolNameTrunc.c_str(),
                                      strFieldWidth);
                }

              if ( GateDigi::GetSingleASCIIMask( 17 ) )
                {
                  rayVolName = (*SDC)[ iDigi ]->GetRayleighVolumeName();
                  G4String rayVolNameTrunc = FixedWidthZeroPaddedString(
                                                                        rayVolName, strMaxLen);
                  m_outputFile.write( rayVolNameTrunc.c_str(),
                                      strFieldWidth);
                }
            }
        }
    }
}

/*!
 * \brief Truncates or pads a string with '\0' for a fixed size
 *
 * Creates a string of a fixed width by truncating it down to the fixed size if
 * necessary, or padding it with a null terminator character, '\0'. This is used
 * to create strings of a fixed width so the event size written out by
 * GateToBinary is fixed.
 *
 * \param full The full string to be referenced
 * \param length The fixed width of the string to be returned.
 */
G4String GateToBinary::FixedWidthZeroPaddedString(const G4String & full, size_t length) {
  G4String trunc = full.substr(0, length);
  trunc += std::string(length - trunc.size(), '\0');
  return (trunc);
}

#endif
