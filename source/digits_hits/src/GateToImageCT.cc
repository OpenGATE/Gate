/*----------------------
  OpenGATE Collaboration

  Didier Benoit <benoit@cppm.in2p3.fr>
  Franca Cassol Brunner <cassol@cppm.in2p3.fr>

  Copyright (C) 2009 imXgam/CNRS, CPPM Marseille

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include <cmath>
#include <vector>

#include "G4Run.hh"
#include "G4VProcess.hh"
#include "G4UnitsTable.hh"
#include "GateRunManager.hh"
#include "G4EmCalculator.hh"
#include "G4TouchableHistory.hh"
#include "G4TransportationManager.hh"

#include "GateOutputMgr.hh"
#include "GateToImageCT.hh"
#include "GateImageCT.hh"
#include "GateToImageCTMessenger.hh"
#include "GateArrayComponent.hh"
#include "GateVSystem.hh"

GateToImageCT::GateToImageCT( const G4String& name, GateOutputMgr* outputMgr,
                              GateVSystem* itsSystem, DigiMode digiMode )
  : GateVOutputModule( name, outputMgr, digiMode )
{
  m_system = itsSystem;

  //value by default
  m_seed = -1;
  m_vrtFactor = 0;
  m_fileName = " "; // All default output file from all output modules are set to " ".
  // They are then checked in GateApplicationMgr::StartDAQ, using
  // the VOutputModule pure virtual method GiveNameOfFile()
  m_inputDataChannel = "Singles";
  m_fastPixelXNb = 0;
  m_fastPixelYNb = 0;
  m_fastPixelZNb = 0;
  m_detectorInX = 0;
  m_detectorInY = 0;
  m_selfDigi = false;

  SetVerboseLevel( 0 );
  m_isEnabled = false; // Keep this flag false: all output are disabled by default

  m_gateImageCT = new GateImageCT();
  m_messenger = new GateToImageCTMessenger( this );
}

GateToImageCT::~GateToImageCT()
{
  delete m_messenger;
  delete m_gateImageCT;
}

const G4String& GateToImageCT::GiveNameOfFile()
{
  return m_fileName;
}

void GateToImageCT::SetFileName( G4String& fileName )
{
  m_fileName = fileName;
}

void GateToImageCT::SetStartSeed( G4int seed )
{
  m_seed = seed;
}

void GateToImageCT::SetVRTFactor( G4int vrtFactor )
{
  m_vrtFactor = vrtFactor;
}

void GateToImageCT::SetFastPixelXNb( G4int fastPixelX )
{
  m_fastPixelXNb = fastPixelX;
}

void GateToImageCT::SetFastPixelYNb( G4int fastPixelY )
{
  m_fastPixelYNb = fastPixelY;
}

void GateToImageCT::SetFastPixelZNb( G4int fastPixelZ )
{
  m_fastPixelZNb = fastPixelZ;
}

void GateToImageCT::SetDetectorX( G4double detectorInX )
{
  m_detectorInX = detectorInX;
}

void GateToImageCT::SetDetectorY( G4double detectorInY )
{
  m_detectorInY = detectorInY;
}

void GateToImageCT::SetSourceDetector( G4double sourceDetector )
{
  m_sourceDetector = sourceDetector;
}

void GateToImageCT::ModuleGeometry()
{
  GateArrayComponent* moduleComponent =
    m_system->FindArrayComponent( "module" );

  //X, Y, Z
  lenghtOfModuleByAxis.reserve( 3 );
  for( G4int i = 0; i != 3; ++i )
    lenghtOfModuleByAxis.push_back( moduleComponent->GetBoxLength( i )  );

  //X, Y, Z
  numberOfModuleByAxis.reserve( 3 );
  for( G4int i = 0; i != 3; ++i )
    numberOfModuleByAxis.push_back(
                                   moduleComponent->GetRepeatNumber( i )  );

  if( numberOfModuleByAxis[0] > 1 || numberOfModuleByAxis[ 2 ] > 1 )
    {
      G4cerr << "[GateToImageCT::RecordBeginOfAcquisition]: \n";
      G4cerr << "The module repeater vector must be parallel to the Y axis"
             << Gateendl;
      G4Exception(
                  "GateToImageCT::ModuleGeometry", "ModuleGeometry", FatalException,"You must change these parameters then restart the simulation" );
    }

  if( nVerboseLevel > 0 )
    {
      G4cout << Gateendl;
      G4cout << "****************************************************"
             << Gateendl;
      G4cout << "**** Dimension of the module in your CT-scanner ****"
             << Gateendl;
      G4cout << "****************************************************"
             << Gateendl;
      G4cout << "size of the module in X : "
             << G4BestUnit(
                           lenghtOfModuleByAxis[ 0 ], "Length" ) << Gateendl;
      G4cout << "size of the module in Y : "
             << G4BestUnit(
                           lenghtOfModuleByAxis[ 1 ], "Length" ) << Gateendl;
      G4cout << "size of the module in Z : "
             << G4BestUnit(
                           lenghtOfModuleByAxis[ 2 ], "Length" ) << Gateendl;
      G4cout << "number of the module in X : "
             << numberOfModuleByAxis[ 0 ] << Gateendl;
      G4cout << "number of the module in Y : "
             << numberOfModuleByAxis[ 1 ] << Gateendl;
      G4cout << "number of the module in Z : "
             << numberOfModuleByAxis[ 2 ] << Gateendl;
    }
}

void GateToImageCT::PixelGeometry( const G4String clusterVolumeArray[],
                                   const G4String pixelVolumeArray[] )
{
  //reserve memory for each vector
  //X, Y, Z, number of pixel by module
  lenghtOfPixelByAxis.reserve( 9 );
  numberOfPixelByAxis.reserve( 9 );
  numberOfPixelByCluster.reserve( 9 );
  numberOfClusterByAxis.reserve( 9 );

  for( G4int ptr = 0; ptr != 3 ; ++ptr )
    {
      GateArrayComponent* pixelComponent =
        m_system->FindArrayComponent( *(pixelVolumeArray + ptr ) );
      GateArrayComponent* clusterComponent =
        m_system->FindArrayComponent( *( clusterVolumeArray + ptr ) );

      for( G4int i = 0; i != 3; ++i )
        {
          lenghtOfPixelByAxis.push_back( pixelComponent->GetBoxLength( i )  );
          numberOfPixelByCluster.push_back(
                                           pixelComponent->GetRepeatNumber( i ) );
          numberOfClusterByAxis.push_back(
                                          clusterComponent->GetRepeatNumber( i ) );
          numberOfPixelByAxis.push_back( numberOfPixelByCluster[ i + 3 * ptr ]
                                         * numberOfClusterByAxis[ i + 3 * ptr ] );
        }

      if( ptr > 0 )
        {
          if( lenghtOfPixelByAxis[ ptr * 3 ] == 0
              || lenghtOfPixelByAxis[ ptr * 3 + 1 ] == 0
              || lenghtOfPixelByAxis[ ptr * 3 + 2 ] == 0 )
            {
              numberOfPixelByAxis[ ptr * 3 ] = 0;
              numberOfPixelByAxis[ ptr * 3 + 1 ] = 0;
              numberOfPixelByAxis[ ptr * 3 + 2 ] = 0;
            }
        }

      if( nVerboseLevel > 0 )
        {
          G4cout << Gateendl;
          G4cout << "*****************************************************"
                 << Gateendl;
          G4cout << "**** Dimension of the " << *(pixelVolumeArray + ptr )
                 << " in your CT-scanner ****\n";
          G4cout << "*****************************************************"
                 << Gateendl;
          G4cout << "size of the " << *(pixelVolumeArray + ptr ) << " in X : "
                 << G4BestUnit(
                               lenghtOfPixelByAxis[ ptr * 3 ], "Length" ) << Gateendl;
          G4cout << "size of the " << *(pixelVolumeArray + ptr ) << " in Y : "
                 << G4BestUnit(
                               lenghtOfPixelByAxis[ ptr * 3 + 1 ], "Length" ) << Gateendl;
          G4cout << "size of the " << *(pixelVolumeArray + ptr ) << " in Z : "
                 << G4BestUnit(
                               lenghtOfPixelByAxis[ ptr * 3 + 2 ], "Length" ) << Gateendl;
          G4cout << "number of the " << *(pixelVolumeArray + ptr )
                 << " in X by module : "
                 << numberOfPixelByAxis[ ptr * 3 ] << Gateendl;
          G4cout << "number of the " << *(pixelVolumeArray + ptr )
                 << " in Y by module : "
                 << numberOfPixelByAxis[ ptr * 3 + 1 ] << Gateendl;
          G4cout << "number of the " << *(pixelVolumeArray + ptr )
                 << " in Z by module : "
                 << numberOfPixelByAxis[ ptr * 3 + 2 ] << Gateendl;
          G4cout << "number of the " << *(pixelVolumeArray + ptr )
                 << " in X by cluster : "
                 << numberOfPixelByCluster[ ptr * 3 ] << Gateendl;
          G4cout << "number of the " << *(pixelVolumeArray + ptr )
                 << " in Y by cluster : "
                 << numberOfPixelByCluster[ ptr * 3 + 1 ] << Gateendl;
          G4cout << "number of the " << *(pixelVolumeArray + ptr )
                 << " in Z by cluster : "
                 << numberOfPixelByCluster[ ptr * 3 + 2 ] << Gateendl;
        }
    }
}

void GateToImageCT::RecordBeginOfAcquisition()
{
  if( nVerboseLevel > 1 )
    G4cout << ">> entering [GateToImageCT::RecordBeginOfAcquisition]"
           << Gateendl;

  //Seed the random
  if(m_seed != -1) {
    CLHEP::HepRandom::setTheSeed( m_seed );
  }

  //Define the number of the first frame, and check the multiplicity of the
  //frame
  G4double fsliceNumber = GetTotalDuration() / GetFrameDuration();
  if ( fabs(fsliceNumber -rint(fsliceNumber)) >= 1.e-5 )
    {
      G4cerr << "[GateToImageCT::RecordBeginOfAcquisition] : \n"
             << "The study duration ( "
             << G4BestUnit( GetTotalDuration(), "Time" )
             << ") does not seem to be a multiple of the time-slice ( "
             << G4BestUnit( GetFrameDuration(), "Time" ) << ").\n";
      G4Exception(
                  "GateToImageCT::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must change these parameters then restart the simulation\n" );
    }

  //store cluster in a array of const G4String
  const G4String clusterVolumeArray[ 3 ] =
    { "cluster_0", "cluster_1", "cluster_2" };

  //store pixel in a array of const G4String
  const G4String pixelVolumeArray[ 3 ] =
    { "pixel_0", "pixel_1", "pixel_2" };

  ModuleGeometry();
  //fast simulation : the user wants a self-made pixelisation
  if( m_fastPixelXNb != 0 && m_fastPixelYNb != 0 )
    {
      G4cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n";
      G4cout << "!!!!! Self-made digitalisation !!!!! \n";
      G4cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n";
      G4cout << Gateendl;

      G4cout << "WARNING : VRT not activate in self-made digitalisation mode"
             << Gateendl;

      m_selfDigi = true;
      std::vector<size_t> fastPixelByAxis( 9, 0 );

      fastPixelByAxis[ 0 ] = m_fastPixelXNb;
      fastPixelByAxis[ 1 ] = m_fastPixelYNb;
      fastPixelByAxis[ 2 ] = m_fastPixelZNb;

      // Prepare the image
      m_gateImageCT->Reset( numberOfModuleByAxis, fastPixelByAxis );
    }
  else
    {
      //Fetche the geometry of the detector and its characteristics
      PixelGeometry( clusterVolumeArray, pixelVolumeArray );
      // Prepare the image
      m_gateImageCT->Reset( numberOfModuleByAxis, numberOfPixelByAxis );
    }

  if( m_vrtFactor != 0 )
    G4cout << Gateendl
           << "--> You asked for a self propagation with multiplicity: "
           << m_vrtFactor << Gateendl;

  if( nVerboseLevel > 1 )
    G4cout << ">> leaving [GateToImageCT::RecordBeginOfAcquisition]"
           << Gateendl;
}

void GateToImageCT::RecordEndOfAcquisition()
{
}

void GateToImageCT::RecordBeginOfRun( const G4Run* aRun )
{
  if( nVerboseLevel > 1 )
    G4cout << " >> entering [GateToImageCT::RecordBeginOfRun]\n";

  G4cout << "#### FrameID = " << GetFrameID() + aRun->GetRunID()
         << " ####\n";

  //One frame per RUN
  m_gateImageCT->ClearData( aRun->GetRunID() );

  if( nVerboseLevel > 1 )
    G4cout << " >> leaving [GateToImageCT::RecordBeginOfRun]\n";
}

void GateToImageCT::RecordEndOfRun( const G4Run* aRun )
{
  if( nVerboseLevel > 1 )
    G4cout << " >> entering [GateToImageCT::RecordEndOfRun]\n";

  // Write the projection sets
  std::ostringstream frameNb;
  frameNb << std::setw( 3 ) << std::setfill( '0' )
          << GetFrameID() + aRun->GetRunID();

  G4String frameFileName;
  frameFileName = m_fileName + "_" + frameNb.str() + ".dat" ;

  std::ofstream m_dataFile;
  m_dataFile.open( frameFileName.c_str(),std::ios::out | std::ios::trunc |
                   std::ios::binary );

  if( !m_dataFile )
    {
      G4cerr
        << "you are in GateToImageCT::RecordEndOfRun( const G4Run* aRun )"
        << Gateendl;
      G4Exception( "GateToImageCT::RecordEndOfRun", "RecordEndOfRun", FatalException, "probleme creating your output data file" );
    }

  m_gateImageCT->StreamOut( m_dataFile );

  G4cout << "--> Image written to the raw file " << frameFileName << Gateendl;

  if( nVerboseLevel > 1 )
    G4cout << " >> leaving [GateToImageCT::RecordEndOfRun]\n";
}

void GateToImageCT::RecordBeginOfEvent( const G4Event* aEvent )
{
  if( nVerboseLevel > 3 )
    G4cout << " >> entering [GateToImageCT::RecordBeginOfEvent]\n";

  if( !aEvent->GetPrimaryVertex()
      || m_detectorInX == 0 || m_detectorInY == 0 )
    return;

  G4ThreeVector direction =
    aEvent->GetPrimaryVertex()->GetPrimary()->GetMomentum();
  //Normalize the momentum
  direction /= sqrt( pow( direction.getX(), 2 ) + pow( direction.getY(), 2 )
                     + pow( direction.getZ(), 2 ) );

  G4ThreeVector position = aEvent->GetPrimaryVertex()->GetPosition();
  G4ThreeVector newPosition = position + direction
    * ( m_sourceDetector / direction.getZ() );

  if( fabs( newPosition.getX() ) > m_detectorInX / 2
      || fabs( newPosition.getY() )  > m_detectorInY / 2 )
    {
      GateRunManager::GetRunManager()->AbortEvent();
      if ( nVerboseLevel > 1 )
        G4cout << " Abort event: Out of detector section "<< Gateendl;
    }

  if( nVerboseLevel > 3 )
    G4cout << " >> leaving [GateToImageCT::RecordBeginOfEvent]\n";
}

void GateToImageCT::RecordEndOfEvent( const G4Event* )
{
  //aEvent = 0;

  const GateDigiCollection* CDS = GetOutputMgr()->
    GetSingleDigiCollection( m_inputDataChannel );

  if( !CDS )
    return;

  if( nVerboseLevel > 3 )
    G4cout << " >> entering [GateToImageCT::RecordEndOfEvent]\n";

  n_digit = CDS->entries();
  for( G4int i = 0; i != n_digit; ++i )
    {
      //find every volumeID
      size_t moduleID = ( (*CDS)[ i ] )->GetComponentID( 1 );
      size_t newPixelID = 0;

      //... if self digitalisation
      if( m_selfDigi )
        {
          //Number of pixel in 1 raw
          m_pixelInRaw = static_cast<size_t>( m_fastPixelXNb );
          //Number of pixel in 1 column
          m_pixelInColumn = static_cast<size_t>( m_fastPixelYNb );

          //Dimension of 1 pixel in X and Y
          G4double widthPixel = lenghtOfModuleByAxis[ 0 ] / m_fastPixelXNb;
          G4double heightPixel = lenghtOfModuleByAxis[ 1 ] / m_fastPixelYNb;

          //Every position > 0
          G4double positionX =
            ( (*CDS)[ i ])->GetLocalPos().x()
            + lenghtOfModuleByAxis[ 0 ] / 2;
          G4double positionY =
            ( (*CDS)[ i ])->GetLocalPos().y()
            + lenghtOfModuleByAxis[ 1 ] / 2;

          G4int raw = static_cast<G4int>( positionY / heightPixel );
          G4int column = static_cast<G4int>( positionX / widthPixel );
          newPixelID = InverseMatrixPixel( column + raw * m_pixelInRaw );

          if( nVerboseLevel > 1 )
            {
              G4cout << "********* Tree VolumeID : \n";
              G4cout << "pixel in X : " << m_pixelInRaw << Gateendl;
              G4cout << "pixel in Y : " << m_pixelInColumn << Gateendl;
              G4cout << "pixelRawID : " << raw << Gateendl;
              G4cout << "pixelColumnID : " << column << Gateendl;
            }
        }
      else
        {
          size_t clusterID = ( (*CDS)[ i ] )->GetComponentID( 2 );
          size_t pixelID = ( (*CDS)[ i ])->GetComponentID( 3 );
          newPixelID = TransformPixel( moduleID, clusterID, pixelID );
        }

      m_gateImageCT->Fill( newPixelID );

      if( nVerboseLevel > 2 )
        {
          G4cout << "newPixelID : " << newPixelID << Gateendl;
          G4cout << Gateendl;
        }
    }

  if( nVerboseLevel > 3 )
    G4cout << " >> leaving [GateToImageCT::RecordEndOfEvent]\n";
}

void GateToImageCT::RecordStepWithVolume( const GateVVolume*,
                                          const G4Step* aStep )
{
  if( nVerboseLevel > 3 )
    G4cout << " >> entering [GateToImageCT::RecordStep]\n";

  if( m_selfDigi )
    return;

  if( m_vrtFactor == 0 )
    return;

  //return if outOfWorld or not on boundary
  if( aStep->GetTrack()->GetNextVolume() == 0
      || aStep->GetPostStepPoint()->GetStepStatus() != fGeomBoundary )
    return;

  const G4String procNameList[ 2 ] = { "PhotoElectric", "Compton" };

  if( aStep->GetPostStepPoint()->GetTouchable()->GetHistoryDepth() == 4 )
    {
      //kill the track
      aStep->GetTrack()->SetTrackStatus( fKillTrackAndSecondaries );

      //store information
      G4Material* material = aStep->GetPostStepPoint()->GetMaterial();
      G4double density = material->GetDensity();
      G4double energy = aStep->GetPostStepPoint()->GetKineticEnergy();
      G4ParticleDefinition* particle =
        aStep->GetTrack()->GetDynamicParticle()->GetDefinition();
      G4ThreeVector momentum = aStep->GetTrack()->GetMomentumDirection();

      if( nVerboseLevel > 0 )
        {
          G4cout << "########################### : NEW PHOTON\n";
          G4cout << "Accelerated factor : " << m_vrtFactor << Gateendl;
        }

      if( nVerboseLevel > 1 )
        {
          G4cout << "density : " << G4BestUnit( density, "Volumic Mass" )
                 << Gateendl;
          G4cout << "energy : " << G4BestUnit( energy, "Energy" ) << Gateendl;
          G4cout << "momentum : " << momentum << Gateendl;
        }

      if( nVerboseLevel > 2 )
        G4cout << "material : " << material << Gateendl;

      //standard energy model
      G4EmCalculator emCalculator;
      G4double sumOfCrossSection = 0.0;
      G4double meanFreePath = 0.0;

      for( G4int i = 0; i != 2 ; ++i )
        {
          G4double massSigma = emCalculator.ComputeCrossSectionPerVolume(
                                                                         energy, particle, procNameList[ i ], material ) / density;
          sumOfCrossSection += massSigma;
        }

      meanFreePath = 1 / sumOfCrossSection / density;

      if( nVerboseLevel > 0 )
        G4cout << "mean free path : "
               << G4BestUnit( meanFreePath, "Length" )
               << Gateendl;

      G4Navigator* theNavigator =
        G4TransportationManager::GetTransportationManager()
        ->GetNavigatorForTracking();

      G4StepPoint* newStepPoint = aStep->GetPostStepPoint();

      //  For all processes except transportation, we select the PostStepPoint volume
      //  For the transportation, we select the PreStepPoint volume
      const G4TouchableHistory* touchable;
      touchable = (const G4TouchableHistory*)(newStepPoint->GetTouchable() );

      GateVolumeID volumeID(touchable);
      if (volumeID.IsInvalid())
        G4Exception(
                    "GateToImageCT::RecordStepWithVolume", "RecordStepWithVolume", FatalException, "could not get the volume ID! Aborting!\n");

      GateOutputVolumeID outputVolumeID =
        m_system->ComputeOutputVolumeID(volumeID);
      size_t moduleID = outputVolumeID[ 1 ];
      size_t clusterID = outputVolumeID[ 2 ];
      size_t pixelID = outputVolumeID[ 3 ];

      G4ThreeVector pos = aStep->GetPostStepPoint()->GetPosition();

      //find the normal initial vector
      G4ThreeVector initialNormalVector = m_system->
        GetBaseComponent()->GetCurrentTranslation();

      //normalise this vector
      G4double lengthVector = initialNormalVector.getR();
      G4ThreeVector initialNormalVectorNormalized =
        initialNormalVector / lengthVector;

      G4RotationMatrix matrixRotation = theNavigator->NetRotation();

      G4ThreeVector normalVector =
        matrixRotation * initialNormalVectorNormalized;

      G4double cosinus = momentum.dot( normalVector );
      G4double geomlimit = lenghtOfPixelByAxis[ 2 ] / cosinus;

      if( nVerboseLevel > 0 )
        G4cout << "geomlimit : " << G4BestUnit( geomlimit, "Length" ) << Gateendl;

      //find the new position in the exit of the detector
      G4SteppingManager* limitSteppingManager = G4EventManager::GetEventManager()
        ->GetTrackingManager()->GetSteppingManager();
      G4Navigator* navigator=limitSteppingManager->GetfNavigator();
      G4double presafety = 0.;
      G4double nextStep =navigator ->CheckNextStep(
                                                   pos,
                                                   momentum,
                                                   10 * lenghtOfPixelByAxis[ 0 ],
                                                   presafety);

      if( nVerboseLevel > 1 )
        G4cout << "nextStep : "
               << G4BestUnit( nextStep, "Length" ) << Gateendl;

      for( G4int i = 0; i != m_vrtFactor; ++i )
        {
          G4double path = 0.;
          // traveled distance
          path = -log( 1 - G4UniformRand() ) * meanFreePath;

          if( nVerboseLevel > 1 )
            G4cout << "path : " << G4BestUnit( path, "Length" ) << Gateendl;

          if( path < geomlimit )
            {
              if( path > nextStep  )
                {
                  G4ThreeVector posStep = ( path * momentum + pos );
                  G4Step* newStep = (G4Step*)aStep;

                  newStep->GetPostStepPoint()->SetPosition( posStep );

                  const G4TouchableHistory* newTouchable;
                  newTouchable = (const G4TouchableHistory*)( newStep->GetPostStepPoint()->GetTouchable() );

                  GateVolumeID VolumeID(newTouchable);

                  GateOutputVolumeID outputVolumeID =
                    m_system->ComputeOutputVolumeID(VolumeID);
                  pixelID = outputVolumeID[ 3 ];
                }
              G4int new_pixelID = TransformPixel(
                                                 moduleID, clusterID, pixelID);
              m_gateImageCT->Fill( new_pixelID );
            }
        }
    }

  if( nVerboseLevel > 3 )
    G4cout << " >> leaving [GateToImageCT::RecordStep]\n";
}

size_t GateToImageCT::InverseMatrixPixel( size_t pixel )
{
  size_t rawID = pixel / m_pixelInRaw ;
  size_t columnID = pixel - m_pixelInRaw * rawID;
  size_t newRawID = m_pixelInColumn - rawID - 1;
  size_t newPixelID = columnID + newRawID * m_pixelInRaw;
  return newPixelID;
}

size_t GateToImageCT::TransformPixel( size_t module,
                                      size_t cluster, size_t pixel )
{
  //find the real ID of pixel in your detector
  size_t pixelRawID = 0;
  size_t newPixelID = 0;
  size_t pixelColumnID = 0;
  size_t numberOfPixelByModule = m_gateImageCT->GetNumberOfPixelByModule();
  m_pixelInRaw = numberOfPixelByAxis[ 0 ] + numberOfPixelByAxis[ 3 ]
    + numberOfPixelByAxis[ 6 ];
  m_pixelInColumn = numberOfPixelByAxis[ 1 ] *
    m_gateImageCT->GetMultiplicityModule();

  if( cluster < numberOfClusterByAxis[ 0 ] )
    {
      pixelRawID = pixel / numberOfPixelByCluster[ 0 ];
      pixelColumnID = pixel - numberOfPixelByCluster[ 0 ] * pixelRawID;
      newPixelID = pixelColumnID + cluster *
        ( numberOfPixelByCluster[ 0 ] + numberOfPixelByAxis[ 3 ]
          + numberOfPixelByAxis[ 6 ] ) + numberOfPixelByModule
        * module + m_pixelInRaw * pixelRawID;
    }
  else if( cluster >= numberOfClusterByAxis[ 0 ]
           && cluster < numberOfClusterByAxis[ 3 ] + numberOfClusterByAxis[ 0 ] )
    {
      pixelRawID = pixel / numberOfPixelByCluster[ 3 ];
      pixelColumnID = pixel - numberOfPixelByCluster[ 3 ] * pixelRawID;
      newPixelID = pixelColumnID + numberOfPixelByCluster[ 0 ]
        + ( cluster - numberOfClusterByAxis[ 0 ] )
        * ( numberOfPixelByCluster[ 3 ]
            + numberOfPixelByCluster[ 6 ] ) + numberOfPixelByModule
        * module + m_pixelInRaw * pixelRawID;
    }
  else
    {
      pixelRawID = pixel / numberOfPixelByCluster[ 6 ];
      pixelColumnID = pixel - numberOfPixelByCluster[ 6 ] * pixelRawID;
      newPixelID = pixelColumnID + numberOfPixelByCluster[ 0 ]
        + numberOfPixelByCluster[ 3 ]+
        + ( cluster - numberOfClusterByAxis[ 0 ]
            - numberOfClusterByAxis[ 3 ] ) * ( numberOfPixelByCluster[ 3 ]
                                               + numberOfPixelByCluster[ 6 ] ) + numberOfPixelByModule
        * module + m_pixelInRaw * pixelRawID;
    }

  if( nVerboseLevel > 1 )
    {
      G4cout << "********* Tree VolumeID : \n";
      G4cout << "pixelID : " << newPixelID << Gateendl;
      G4cout << "moduleID : " << module << Gateendl;
      G4cout << "clusterID : " << cluster << Gateendl;
      G4cout << "pixelRawID : " << pixelRawID << Gateendl;
      G4cout << "pixelColumnID : " << pixelColumnID << Gateendl;
    }

  return  InverseMatrixPixel( newPixelID );
}
