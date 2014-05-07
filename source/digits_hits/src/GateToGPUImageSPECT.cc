/*----------------------
   OpenGATE Collaboration

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/
#include <cmath>
#include "pthread.h"
#include <vector>
#include "G4RunManager.hh"
#include "GateConfiguration.h"
#include "G4VProcess.hh"
#include "G4UnitsTable.hh"
#include "G4RunManager.hh"
#include "G4EmCalculator.hh"
#include "G4TouchableHistory.hh"
#include "G4TransportationManager.hh"
#include "GateSourceMgr.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateOutputMgr.hh"
#include "GateRunManager.hh"
#include "GateArrayComponent.hh"
#include "GateVSystem.hh"
#include "GateToGPUImageSPECT.hh"
#include "GateToGPUImageSPECTMessenger.hh"
#include "G4VUserTrackInformation.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include <cstddef>
#include <sys/time.h>

#define PI ( 4 * ::atan( 1 ) )

GateToGPUImageSPECT::GateToGPUImageSPECT( const G4String& name,
    GateOutputMgr *outputMgr, GateVSystem* itsSystem, DigiMode digiMode )
: GateVOutputModule( name, outputMgr, digiMode ), m_system( itsSystem ),
m_bufferParticleEntry( 1000000 ), m_cudaDevice( 0 ), m_cpuNumber( 0 ),
m_cpuFlag( false ), m_rootHitFlag( false ),
m_rootSingleFlag( false ), m_rootSourceFlag( false ),
m_rootExitCollimatorSourceFlag( false ), m_timeFlag( false ), m_ny_pixel( 0 ),
m_nz_pixel( 0 ), m_centerOfPxlZ( NULL ), m_centerOfPxlY( NULL ),
m_gpuCollimator( NULL ), m_septa( 0.0 ), m_fy( 0.0 ), m_fz( 0.0 ),
m_collimatorHeight( 0.0 ),
m_spaceBetweenCollimatorDetector( 0.0 ), m_ror( 0.0 ),
m_launchLastBuffer( false ), m_isAlreadyLaunchedBuffer( false ),
m_elapsedTime( 0.0 ), m_gpuParticle( NULL )
{
	GateVOutputModule::Enable( false ); // Keep this flag false: all output are disabled by default
	m_messenger = new GateToGPUImageSPECTMessenger( this );
}

GateToGPUImageSPECT::~GateToGPUImageSPECT()
{
	delete m_messenger;
}

static unsigned int binary_search_cpu( float position, float *tab, unsigned int maxid )
{
	unsigned short int begIdx = 0;
	unsigned short int endIdx = maxid - 1;
	unsigned short int medIdx = endIdx / 2;

	while (endIdx-begIdx > 1) {
		if (position < tab[medIdx]) {begIdx = medIdx;}
       else {endIdx = medIdx;}
        medIdx = (begIdx+endIdx) / 2;
    }
    return medIdx;
}

static void* cpuSPECT( void *args )
{
	// Taking arguments
	ThreadArgSPECT *pArgs = (ThreadArgSPECT*)args;
	GateCPUCollimator *cpuCollimator = pArgs->m_cpuCollimator;
	GateCPUParticle *cpuParticle = pArgs->m_cpuParticle;
	unsigned int nThread = pArgs->nThread;
	unsigned int tID = pArgs->tID;

	unsigned int size = cpuParticle->size;

	// Compute the number of particles by thread
	unsigned int *particleThread = new unsigned int[ nThread ];
	unsigned int nParticleByThread = size / nThread;
	unsigned int remParticle = size % nThread;
	for( unsigned int i = 0; i < nThread; ++i )
	{
		particleThread[ i ] = nParticleByThread;
	}
	for( unsigned int i = 0; i < remParticle; ++i )
	{
		particleThread[ i ] += 1;
	}

	// Find the begin and end by thread
	unsigned int begin = 0;
	for( unsigned int i = 0; i < tID; ++i )
	{
		begin += particleThread[ i ];
	}
	unsigned int end = begin + particleThread[ tID ];

	// Taking collimator informations
	float *entry_collim_y = cpuCollimator->entry_collim_y;
	float *entry_collim_z = cpuCollimator->entry_collim_z;
	float *exit_collim_y = cpuCollimator->exit_collim_y;
	float *exit_collim_z = cpuCollimator->exit_collim_z;
	unsigned int y_size = cpuCollimator->y_size;
	unsigned int z_size = cpuCollimator->z_size;
	float planeToProject = cpuCollimator->planeToProject + cpuParticle->px[ 0 ];
	float *hole = cpuParticle->hole;

	// Loop over the particles
	for( unsigned int p = begin; p < end; ++p )
	{
		// Taking particle informations
		float px = cpuParticle->px[ p ];
		float py = cpuParticle->py[ p ];
		float pz = cpuParticle->pz[ p ];
		float dx = cpuParticle->dx[ p ];
		float dy = cpuParticle->dy[ p ];
		float dz = cpuParticle->dz[ p ];

		if( py > entry_collim_y[ 0 ] || py < entry_collim_y[ y_size - 1 ] )
		{
			hole[ p ] = -1.0f;
			continue;
		}

		if( pz > entry_collim_z[ 0 ] || pz < entry_collim_z[ z_size - 1 ] )
		{
			hole[ p ] = -1.0f;
			continue;
		}

		// Map entry
		unsigned int index_entry_y = binary_search_cpu( py, entry_collim_y, y_size );
		unsigned int index_entry_z = binary_search_cpu( pz, entry_collim_z, z_size );
		unsigned char is_in_hole_y = ( index_entry_y & 1 ) ? 0 : 1;
		unsigned char is_in_hole_z = ( index_entry_z & 1 ) ? 0 : 1;
		unsigned char in_hole = is_in_hole_y & is_in_hole_z;
		hole[ p ] = ( in_hole )? index_entry_y * z_size + index_entry_z : -1.0f;

		// Project point
		if( hole[ p ] == -1.0f )
			continue;

		float n[ 3 ] = { -1.0f, 0.0f, 0.0f };
		float v0[ 3 ] = { planeToProject, 0.0f, 0.0f };
		float momentum[ 3 ] = { dx, dy, dz };
		float position[ 3 ] = { px, py, pz };

		float a[ 3 ] = {
			v0[ 0 ] - position[ 0 ],
			v0[ 1 ] - position[ 1 ],
			v0[ 2 ] - position[ 2 ]
		};

		float s2 = n[ 0 ] * momentum[ 0 ] + n[ 1 ] * momentum[ 1 ] + n[ 2 ] * momentum[ 2 ];
		float s1 = n[ 0 ] * a[ 0 ] + n[ 1 ] * a[ 1 ] + n[ 2 ] * a[ 2 ];
		float s = s1 / s2;

		// New position
		px += momentum[ 0 ] * s;
		py += momentum[ 1 ] * s;
		pz += momentum[ 2 ] * s;

		cpuParticle->px[ p ] = px;
		cpuParticle->py[ p ] = py;
		cpuParticle->pz[ p ] = pz;

		if( py > exit_collim_y[ 0 ] || py < exit_collim_y[ y_size - 1 ] )
		{
			hole[ p ] = -1.0f;
			continue;
		}

		if( pz > exit_collim_z[ 0 ] || pz < exit_collim_z[ z_size - 1 ] )
		{
			hole[ p ] = -1.0f;
			continue;
		}

		// Map exit
		unsigned int index_exit_y = binary_search_cpu( py, exit_collim_y, y_size );
		unsigned int index_exit_z = binary_search_cpu( pz, exit_collim_z, z_size );
		is_in_hole_y = ( index_exit_y & 1 )? 0 : 1;
		is_in_hole_z = ( index_exit_z & 1 )? 0 : 1;
		in_hole = is_in_hole_y & is_in_hole_z;
		int newhole = ( in_hole )? index_exit_y * z_size + index_exit_z : -1;
		if( newhole == -1 )
		{
			hole[ p ] = -1.0f;
			continue;
		}
		if( newhole != hole[ p ] )
		{
			hole[ p ] = -1.0f;
		}
	}

	delete[] particleThread;
	pthread_exit( NULL );
}

void GateToGPUImageSPECT::SetRootHitFlag( G4bool flag )
{
    m_rootHitFlag = flag;
}

void GateToGPUImageSPECT::SetRootSingleFlag( G4bool flag )
{
    m_rootSingleFlag = flag;
}

void GateToGPUImageSPECT::SetRootSourceFlag( G4bool flag )
{
    m_rootSourceFlag = flag;
}

void GateToGPUImageSPECT::SetRootExitCollimatorSourceFlag( G4bool flag )
{
    m_rootExitCollimatorSourceFlag = flag;
}

void GateToGPUImageSPECT::SetTimeFlag( G4bool flag )
{
	m_timeFlag = flag;
}

const G4String& GateToGPUImageSPECT::GiveNameOfFile()
{
  return m_fileName;
}

void GateToGPUImageSPECT::SetFileName( G4String& fileName )
{
	m_fileName = fileName;
}

void GateToGPUImageSPECT::SetBufferParticleEntry( G4int bufferParticleEntry )
{
    m_bufferParticleEntry = bufferParticleEntry;
}

void GateToGPUImageSPECT::SetCudaDevice( G4int cudaDevice )
{
    m_cudaDevice = cudaDevice;
}

void GateToGPUImageSPECT::SetCpuNumber( G4int cpuNumber )
{
	m_cpuNumber = cpuNumber;
}

void GateToGPUImageSPECT::SetCpuFlag( G4bool flag )
{
	m_cpuFlag = flag;
}

void GateToGPUImageSPECT::SetNYpixel( G4int ny )
{
    m_ny_pixel = ny;
}

void GateToGPUImageSPECT::SetNZpixel( G4int nz )
{
    m_nz_pixel = nz;
}

void GateToGPUImageSPECT::SetZPixelSize( G4double zPixelSize )
{
	m_z_pixel_size = zPixelSize;
}

void GateToGPUImageSPECT::SetYPixelSize( G4double yPixelSize )
{
	m_y_pixel_size = yPixelSize;
}

void GateToGPUImageSPECT::SetSepta( G4double septa )
{
    m_septa = septa;
}

void GateToGPUImageSPECT::SetFy( G4double fY )
{
    m_fy = fY;
}

void GateToGPUImageSPECT::SetFz( G4double fZ )
{
    m_fz = fZ;
}

void GateToGPUImageSPECT::SetCollimatorHeight( G4double collimatorHeight )
{
    m_collimatorHeight = collimatorHeight;
}

void GateToGPUImageSPECT::SetSpaceBetweenCollimatorDetector( G4double spaceBetweenCollimatorDetector )
{
    m_spaceBetweenCollimatorDetector = spaceBetweenCollimatorDetector;
}

void GateToGPUImageSPECT::SetRor( G4double ror )
{
	m_ror = ror;
}

void GateToGPUImageSPECT::RecordBeginOfAcquisition()
{
	if( nVerboseLevel > 1 )
		G4cout << ">> entering [GateToGPUImageSPECT::RecordBeginOfAcquisition]"
			   << G4endl;

    // Creating ROOT file
    G4cout << "GateToGPUImageSPECT: ROOT files creating..." << G4endl;
    m_file = new TFile( ( m_fileName + ".root" ).c_str(), "RECREATE" );

    // Check that we succeeded in opening the file
    if( !m_file )
	{
        G4String msg = "Could not open the requested output ROOT file '"
            + m_fileName + ".root'!";
        G4Exception( "GateToGPUImageSPECT::RecordBeginOfAcquisition",
            "RecordBeginOfAcquisition", FatalException, msg );
	}
		if ( !(m_file->IsOpen()) )
	{
		G4String msg = "Could not open the requested output ROOT file '"
            + m_fileName + ".root'!";
        G4Exception( "GateToGPUImageSPECT::RecordBeginOfAcquisition",
            "RecordBeginOfAcquisition", FatalException, msg );
	}

    if( nVerboseLevel > 0 )
        G4cout << "GateToGPUImageSPECT::RecordBeginOfAcquisition, filename: "
            << m_fileName << ".root created" << G4endl;

		if( m_cpuFlag )
			m_cpuParticle = GateCPUParticle_new( m_bufferParticleEntry );
		else
    	m_gpuParticle = GateGPUParticle_new( m_bufferParticleEntry, m_cudaDevice );

    // Creating TTree
    GPUHits =    new TTree( G4String( "GPU_HITS" ).c_str(), "GPUHITS" );
    GPUSingles = new TTree( G4String( "GPU_SINGLES" ).c_str(), "GPUSINGLES" );
    GPUSource =  new TTree( G4String( "GPU_SOURCE" ).c_str(), "GPUSOURCE" );
		GPUExitCollimatorSource = new TTree(
			G4String( "GPU_EXIT_COLLIMATOR_SOURCE" ).c_str(),
			"GPUEXITCOLLIMATORSOURCE" );

    // Creating source branches
    GPUSource->Branch( G4String( "sourcePosX" ).c_str(), &m_posX_Source,
        "sourcePosX/F" );
    GPUSource->Branch( G4String( "sourcePosY" ).c_str(), &m_posY_Source,
        "sourcePosY/F" );
    GPUSource->Branch( G4String( "sourcePosZ" ).c_str(), &m_posZ_Source,
        "sourcePosZ/F" );
		GPUSource->Branch( G4String( "energy" ).c_str(), &m_energy_Source,
        "energy/F" );
		GPUSource->Branch( G4String( "runID" ).c_str(), &m_runID_Source,
        "runID/I" );

    // Creating hits branches
    GPUHits->Branch( G4String( "posX" ).c_str(), &m_posX_Hit, "posX/F" );
    GPUHits->Branch( G4String( "posY" ).c_str(), &m_posY_Hit, "posY/F" );
    GPUHits->Branch( G4String( "posZ" ).c_str(), &m_posZ_Hit, "posZ/F" );
    GPUHits->Branch( G4String( "energy" ).c_str(), &m_energy_Hit, "energy/F" );
    GPUHits->Branch( G4String( "time" ).c_str(), &m_time_Hit, "time/F" );
		GPUHits->Branch( G4String( "runID" ).c_str(), &m_runID_Hit, "runID/I" );

    // Creating singles branches
    GPUSingles->Branch( G4String( "posX" ).c_str(), &m_posX_Single, "posX/F" );
    GPUSingles->Branch( G4String( "posY" ).c_str(), &m_posY_Single, "posY/F" );
    GPUSingles->Branch( G4String( "posZ" ).c_str(), &m_posZ_Single, "posZ/F" );
    GPUSingles->Branch( G4String( "energy" ).c_str(), &m_energy_Single,
        "energy/F" );
    GPUSingles->Branch( G4String( "time" ).c_str(), &m_time_Single, "time/F" );
		GPUSingles->Branch( G4String( "runID" ).c_str(), &m_runID_Single, "runID/I" );
		GPUSingles->Branch( G4String( "sourcePosX" ).c_str(), &m_src_posX_Single, "sourcePosX/F" );
		GPUSingles->Branch( G4String( "sourcePosY" ).c_str(), &m_src_posY_Single, "sourcePosY/F" );
		GPUSingles->Branch( G4String( "sourcePosZ" ).c_str(), &m_src_posZ_Single, "sourcePosZ/F" );

		// Creating exit collimator source branches
    GPUExitCollimatorSource->Branch( G4String( "sourcePosX" ).c_str(),
			&m_posX_ExitCollimatorSource, "sourcePosX/F" );
    GPUExitCollimatorSource->Branch( G4String( "sourcePosY" ).c_str(),
			&m_posY_ExitCollimatorSource, "sourcePosY/F" );
    GPUExitCollimatorSource->Branch( G4String( "sourcePosZ" ).c_str(),
			&m_posZ_ExitCollimatorSource, "sourcePosZ/F" );
		GPUExitCollimatorSource->Branch( G4String( "energy" ).c_str(),
			&m_energy_ExitCollimatorSource, "energy/F" );
		GPUExitCollimatorSource->Branch( G4String( "runID" ).c_str(),
			&m_runID_ExitCollimatorSource, "runID/I" );

    // Getting detector geometry
/*    G4double y_pixel_size = m_system->FindComponent( "pixel" )
        ->GetCreator()->GetHalfDimension( 1 )/mm * 2.0;
    G4double z_pixel_size = m_system->FindComponent( "pixel" )
        ->GetCreator()->GetHalfDimension( 2 )/mm * 2.0;
*/

    // Computing first pixel in Z and Y
    m_centerOfPxlZ = new G4double[ m_nz_pixel ];
    m_centerOfPxlY = new G4double[ m_ny_pixel ];
    // Filling buffer
    // In Z
    for( G4int i = 0; i < m_nz_pixel; ++i )
    {
        m_centerOfPxlZ[ i ] =
            ( ( ( m_nz_pixel - 1.0 ) / 2.0 ) - i ) * m_z_pixel_size;
    }
    // In Y
    for( G4int i = 0; i < m_ny_pixel; ++i )
    {
        m_centerOfPxlY[ i ] =
            ( ( ( m_ny_pixel - 1.0 ) / 2.0 ) - i ) * m_y_pixel_size;
    }

    // Allocating memory for the collimator
		if( m_cpuFlag )
		{
			if( nVerboseLevel>0 )
				G4cout << "Creating CPU collimator..." << G4endl;

			m_cpuCollimator = GateCPUCollimator_new( m_ny_pixel, m_nz_pixel, m_septa,
				m_fy, m_fz, m_collimatorHeight,
				m_spaceBetweenCollimatorDetector, m_centerOfPxlY, m_centerOfPxlZ,
				m_y_pixel_size, m_z_pixel_size );
		}
		#ifdef GATE_USE_GPU
		else
		{
			if( nVerboseLevel>0 )
				G4cout << "Creating GPU collimator..." << G4endl;

			m_gpuCollimator = GateGPUCollimator_new( m_ny_pixel, m_nz_pixel, m_septa,
        m_fy, m_fz, m_collimatorHeight,
        m_spaceBetweenCollimatorDetector, m_centerOfPxlY, m_centerOfPxlZ,
        m_y_pixel_size, m_z_pixel_size, m_cudaDevice );
				GateGPUCollimator_init( m_gpuCollimator );
		}
		#endif


	if( nVerboseLevel > 1 )
		G4cout << ">> leaving [GateToGPUImageSPECT::RecordBeginOfAcquisition]"
			<< G4endl;
}

void GateToGPUImageSPECT::RecordEndOfAcquisition()
{
    if( nVerboseLevel > 1 )
		G4cout << ">> entering [GateToGPUImageSPECT::RecordEndOfAcquisition]"
			   << G4endl;

    // Freeing memory
		if( m_cpuFlag )
		{
			if( nVerboseLevel>0 )
				G4cout << "Deleting CPU collimator..." << G4endl;

    	GateCPUCollimator_delete( m_cpuCollimator );
    	GateCPUParticle_delete( m_cpuParticle );
		}
		else
		{
			if( nVerboseLevel>0 )
				G4cout << "Deleting GPU collimator..." << G4endl;

    	GateGPUCollimator_delete( m_gpuCollimator );
    	GateGPUParticle_delete( m_gpuParticle );
		}

    m_file->Write();
    G4cout << "GateToGPUImageSPECT: ROOT files closing..." << G4endl;
    if( m_file->IsOpen() )
    {
        m_file->Close();
    }

		if( m_timeFlag )
		{
			//G4cout << "Elapsed time in GPU/SPECT collimator: " << m_elapsedTime
			//	<< " seconds" << G4endl;
            printf("Elapsed time in SPECT collimator: %f s\n", m_elapsedTime);
		}

	if( nVerboseLevel > 1 )
		G4cout << ">> leaving [GateToGPUImageSPECT::RecordEndOfAcquisition]"
			<< G4endl;
}

void GateToGPUImageSPECT::RecordBeginOfRun( const G4Run* aRun )
{
	if( nVerboseLevel > 1 )
		G4cout << " >> entering [GateToGPUImageSPECT::RecordBeginOfRun]" << G4endl;

	m_runID = aRun->GetRunID();

	m_launchLastBuffer = false;
	m_isAlreadyLaunchedBuffer= false;

	G4double timeStart = GateApplicationMgr::GetInstance()->GetTimeStart()
		+ m_runID * GateApplicationMgr::GetInstance()->GetTimeSlice();

	G4double velocity = m_system->GetBaseComponent()->GetOrbitingVelocity();

	m_angle = ( velocity * timeStart )/deg;


	if( nVerboseLevel > 1 )
		G4cout << " >> leaving [GateToGPUImageSPECT::RecordBeginOfRun]" << G4endl;
}

void GateToGPUImageSPECT::RecordEndOfRun( const G4Run* )
{
	if( nVerboseLevel > 1 )
		G4cout << " >> entering [GateToGPUImageSPECT::RecordEndOfRun]" << G4endl;

	if( nVerboseLevel > 1 )
		G4cout << " >> leaving [GateToGPUImageSPECT::RecordEndOfRun]" << G4endl;
}

void GateToGPUImageSPECT::RecordBeginOfEvent( const G4Event* aEvent )
{
	if( nVerboseLevel > 3 )
		G4cout << " >> entering [GateToGPUImageSPECT::RecordBeginOfEvent]" << G4endl;

	if( !aEvent->GetPrimaryVertex() )
		return;

    // Storing the point emission source
    if( m_rootSourceFlag )
    {
        G4ThreeVector positionVertex = aEvent->GetPrimaryVertex()->GetPosition();
        m_posX_Source = (G4float)positionVertex.x()/mm;
        m_posY_Source = (G4float)positionVertex.y()/mm;
        m_posZ_Source = (G4float)positionVertex.z()/mm;
				m_energy_Source = (G4float)aEvent->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy();
				m_runID_Source = m_runID;
        GPUSource->Fill();
    }

	m_launchLastBuffer = GateSourceMgr::GetInstance()->IsLaunchLastBuffer();

	if( nVerboseLevel > 3 )
		G4cout << " >> leaving [GateToGPUImageSPECT::RecordBeginOfEvent]" << G4endl;
}

void GateToGPUImageSPECT::RecordEndOfEvent( const G4Event* )
{
	if( nVerboseLevel > 3 )
		G4cout << " >> entering [GateToGPUImageSPECT::RecordEndOfEvent]" << G4endl;

    // Tracking singles
    if( m_rootSingleFlag )
    {
        // Taking the single digit collection
        GateSingleDigiCollection const* CDS = GetOutputMgr()->
		    GetSingleDigiCollection( "Singles" );

        if( !CDS ) return;

        // Loop over the singles
        G4int singles = CDS->entries();
        for( G4int i = 0; i != singles; ++i )
        {
            m_energy_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetEnergy()/keV;
            m_posX_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetGlobalPos().x()/mm;
            m_posY_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetGlobalPos().y()/mm;
            m_posZ_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetGlobalPos().z()/mm;
            m_time_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetTime()/s;
						m_src_posX_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetSourcePosition().x()/mm;
						m_src_posY_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetSourcePosition().y()/mm;
						m_src_posZ_Single = (G4float)( (*CDS)[ i ]->GetPulse() ).GetSourcePosition().z()/mm;
						m_runID_Single = m_runID;
            GPUSingles->Fill();
        }
    }

    // Tracking hits
    if( m_rootHitFlag )
    {
        // Taking the hit digit collection
        GateCrystalHitsCollection* CHC = GetOutputMgr()->
            GetCrystalHitCollection();

        if( !CHC ) return;

        // Loop over the singles
        G4int hits = CHC->entries();
        for( G4int i = 0; i != hits; ++i )
        {
            if( (*CHC)[i]->GoodForAnalysis() )
            {
                GateCrystalHit* aHit = (*CHC)[i];

                m_energy_Hit = (G4float)aHit->GetEdep()/keV;
                m_posX_Hit = (G4float)aHit->GetGlobalPos().x()/mm;
                m_posY_Hit = (G4float)aHit->GetGlobalPos().y()/mm;
                m_posZ_Hit = (G4float)aHit->GetGlobalPos().z()/mm;
                m_time_Hit = (G4float)aHit->GetTime()/s;
								m_runID_Hit = m_runID;
                GPUHits->Fill();
            }
        }
    }

	if( nVerboseLevel > 3 )
		G4cout << " >> leaving [GateToGPUImageSPECT::RecordEndOfEvent]" << G4endl;
}

void GateToGPUImageSPECT::RecordStepWithVolume( const GateVVolume*,
	const G4Step *aStep )
{
	if( nVerboseLevel > 3 )
		G4cout << " >> entering [GateToGPUImageSPECT::RecordStep]" << G4endl;

	// Filter on new created particle on parentID and PDG
	G4int PDGencoding = aStep->GetTrack()->GetDynamicParticle()->GetPDGcode();

	// Check if we are on the boundary
	G4StepPoint *preStep = aStep->GetPreStepPoint();
	G4String volName = preStep->GetPhysicalVolume()->GetName();

	unsigned int id = 0;
	if( preStep->GetStepStatus() == fGeomBoundary && volName == m_volToAttach
		&& PDGencoding == 22 )
	{
		// Rotate position
		G4double x = preStep->GetPosition().x()/mm;
		G4double y = preStep->GetPosition().y()/mm;
		G4double z = preStep->GetPosition().z()/mm;
		G4double newX = ::cos( m_angle * PI / 180.0 ) * x
			- ::sin( m_angle * PI / 180.0 ) * z;
		G4double newZ = ::sin( m_angle * PI / 180.0 ) * x
			+ ::cos( m_angle * PI / 180.0 ) * z;

		// Check if the particule is on entry on collimator
		if( newX <= ( m_ror + 0.001 ) ) // 0.1 of security double precision
		{
			if( m_cpuFlag )
			{
				m_cpuParticle->size += 1;
				id = m_cpuParticle->size - 1;
			}
			else
			{
				m_gpuParticle->size += 1;
				id = m_gpuParticle->size - 1;
			}

			// Rotate momentum
			G4double dx = preStep->GetMomentumDirection().x();
			G4double dy = preStep->GetMomentumDirection().y();
			G4double dz = preStep->GetMomentumDirection().z();
			G4double newDx = ::cos( m_angle * PI / 180.0 ) * dx
				- ::sin( m_angle * PI / 180.0 ) * dz;
			G4double newDz = ::sin( m_angle * PI / 180.0 ) * dx
				+ ::cos( m_angle * PI / 180.0 ) * dz;

			if( aStep->GetTrack()->GetDefinition() == G4Gamma::Gamma() )
			{
				if( m_cpuFlag )
					m_cpuParticle->type[ id ] = 0;
				else
					m_gpuParticle->type[ id ] = 0;
			}

      if( aStep->GetTrack()->GetDefinition() == G4Electron::Electron() )
			{
				if( m_cpuFlag )
					m_cpuParticle->type[ id ] = 0;
				else
					m_gpuParticle->type[ id ] = 1;
			}

			if( m_cpuFlag )
			{
				m_cpuParticle->E[ id ] = preStep->GetKineticEnergy()/MeV;
				m_cpuParticle->parentID[ id ] = aStep->GetTrack()->GetParentID();
				m_cpuParticle->eventID[ id ] =
				GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
				m_cpuParticle->trackID[ id ] = aStep->GetTrack()->GetTrackID();
				m_cpuParticle->t[ id ] = preStep->GetGlobalTime();
					m_cpuParticle->px[ id ] = newX;
					m_cpuParticle->py[ id ] = y;
					m_cpuParticle->pz[ id ] = newZ;
					m_cpuParticle->dx[ id ] = newDx;
					m_cpuParticle->dy[ id ] = dy;
					m_cpuParticle->dz[ id ] = newDz;
				}
				else
				{
					m_gpuParticle->E[ id ] = preStep->GetKineticEnergy()/MeV;
					m_gpuParticle->parentID[ id ] = aStep->GetTrack()->GetParentID();
					m_gpuParticle->eventID[ id ] =
						GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
					m_gpuParticle->trackID[ id ] = aStep->GetTrack()->GetTrackID();
					m_gpuParticle->t[ id ] = preStep->GetGlobalTime();
					m_gpuParticle->px[ id ] = newX;
					m_gpuParticle->py[ id ] = y;
					m_gpuParticle->pz[ id ] = newZ;
					m_gpuParticle->dx[ id ] = newDx;
					m_gpuParticle->dy[ id ] = dy;
					m_gpuParticle->dz[ id ] = newDz;
				}

				aStep->GetTrack()->SetTrackStatus( fKillTrackAndSecondaries );
			}
		}

	unsigned int limitBuffer = 0;
	if( m_cpuFlag )
		limitBuffer = m_cpuParticle->size;
	else
		limitBuffer = m_gpuParticle->size;

	if( (limitBuffer == m_bufferParticleEntry || m_launchLastBuffer )
			&& !m_isAlreadyLaunchedBuffer )
    {
				if( nVerboseLevel > 0 )
				{
					if( m_cpuFlag )
					{
						G4cout << "->->->->-> Launching CPU on " << m_cpuNumber
							<< " thread(s) ... <-<-<-<-<-" << G4endl;
						G4cout << "Particle entry CPU: " << m_cpuParticle->size << G4endl;
					}
					else
					{
						G4cout << "->->->->-> Launching GPU... <-<-<-<-<-" << G4endl;
						G4cout << "Particle entry GPU: " << m_gpuParticle->size << G4endl;
					}
				}

                // A real timing :)
	            timeval tv;
                G4double start, end;

				unsigned int sizeAfter = 0;
				if( m_cpuFlag )
				{
                    // timing - JB
	                gettimeofday(&tv, NULL);
	                start = tv.tv_sec + tv.tv_usec / 1000000.0;

					pthread_t *threads = new pthread_t[ m_cpuNumber ];
					pthread_attr_t attr;
					pthread_attr_init( &attr );
					pthread_attr_setdetachstate( &attr, PTHREAD_CREATE_JOINABLE );

					ThreadArgSPECT *arg = new ThreadArgSPECT[ m_cpuNumber ];
					for( int t = 0; t < m_cpuNumber; ++t )
					{
						arg[ t ].m_cpuCollimator = m_cpuCollimator;
						arg[ t ].m_cpuParticle = m_cpuParticle;
						arg[ t ].nThread = m_cpuNumber;
						arg[ t ].tID = t;
						pthread_create( &threads[ t ], NULL, cpuSPECT, (void*)&arg[ t ] );
					}
					for( int t = 0; t < m_cpuNumber; ++t )
					{
						pthread_join( threads[ t ], NULL );
					}
					delete[] arg;
					delete[] threads;
					pthread_attr_destroy( &attr );

					// Pack data
					int pack = 0;
					unsigned int i = 0;
					while( i < m_cpuParticle->size )
					{
						if( m_cpuParticle->hole[ i ] == -1.0f )
						{
							i++;
							continue;
						}
						m_cpuParticle->px[ pack ] = m_cpuParticle->px[ i ];
						m_cpuParticle->py[ pack ] = m_cpuParticle->py[ i ];
						m_cpuParticle->pz[ pack ] = m_cpuParticle->pz[ i ];
						m_cpuParticle->dx[ pack ] = m_cpuParticle->dx[ i ];
						m_cpuParticle->dy[ pack ] = m_cpuParticle->dy[ i ];
						m_cpuParticle->dz[ pack ] = m_cpuParticle->dz[ i ];
						m_cpuParticle->eventID[ pack ] = m_cpuParticle->eventID[ i ];
						m_cpuParticle->parentID[ pack ] = m_cpuParticle->parentID[ i ];
						m_cpuParticle->trackID[ pack ] = m_cpuParticle->trackID[ i ];
						m_cpuParticle->t[ pack ] = m_cpuParticle->t[ i ];
						m_cpuParticle->E[ pack ] = m_cpuParticle->E[ i ];
						m_cpuParticle->type[ pack ] = m_cpuParticle->type[ i ];
						pack++, i++;
					}
					m_cpuParticle->size = pack;

                    // timing - JB
	                gettimeofday(&tv, NULL);
	                end = tv.tv_sec + tv.tv_usec / 1000000.0;
					m_elapsedTime += (G4double)(end - start);

					// Getting the number of particles after the CPU
					sizeAfter = m_cpuParticle->size;
					if( nVerboseLevel > 0 )
						G4cout << "Particle exit CPU: " << sizeAfter<< G4endl;
				}
				#ifdef GATE_USE_GPU
				else
				{
					// CPU -> GPU
                    // timing - JB
	                gettimeofday(&tv, NULL);
	                start = tv.tv_sec + tv.tv_usec / 1000000.0;

					GateGPUCollimator_process( m_gpuCollimator, m_gpuParticle );

                    // timing - JB
	                gettimeofday(&tv, NULL);
	                end = tv.tv_sec + tv.tv_usec / 1000000.0;
					m_elapsedTime += (G4double)(end - start);

					// Getting the number of particles after the GPU
					sizeAfter = m_gpuParticle->size;
					if( nVerboseLevel > 0 )
						G4cout << "Particle exit GPU: " << m_gpuParticle->size << G4endl;
				}
				#endif

				// Storing the point emission source
				if( m_rootExitCollimatorSourceFlag )
				{
					if( m_cpuFlag )
					{
						for( unsigned int i = 0; i < m_cpuParticle->size; ++i )
						{
							G4double x = m_cpuParticle->px[ i ] * ::cos( m_angle * PI / 180.0 )
								+ m_cpuParticle->pz[ i ] * ::sin( m_angle * PI / 180.0 );
							G4double y = (G4float)m_cpuParticle->py[ i ];
							G4double z = m_cpuParticle->pz[ i ] * ::cos( m_angle * PI / 180.0 )
								- m_cpuParticle->px[ i ] * ::sin( m_angle * PI / 180.0 );

							m_posX_ExitCollimatorSource = (G4float)x/mm;
							m_posY_ExitCollimatorSource = (G4float)y/mm;
							m_posZ_ExitCollimatorSource = (G4float)z/mm;
							m_energy_ExitCollimatorSource = (G4float)m_cpuParticle->E[ i ]/keV;
							m_runID_ExitCollimatorSource = m_runID;
							GPUExitCollimatorSource->Fill();
						}
					}
					else
					{
						for( unsigned int i = 0; i < m_gpuParticle->size; ++i )
						{
							G4double x = m_gpuParticle->px[ i ] * ::cos( m_angle * PI / 180.0 )
								+ m_gpuParticle->pz[ i ] * ::sin( m_angle * PI / 180.0 );
							G4double y = (G4float)m_gpuParticle->py[ i ];
							G4double z = m_gpuParticle->pz[ i ] * ::cos( m_angle * PI / 180.0 )
								- m_gpuParticle->px[ i ] * ::sin( m_angle * PI / 180.0 );

							m_posX_ExitCollimatorSource = (G4float)x/mm;
							m_posY_ExitCollimatorSource = (G4float)y/mm;
							m_posZ_ExitCollimatorSource = (G4float)z/mm;
							m_energy_ExitCollimatorSource = (G4float)m_gpuParticle->E[ i ]/keV;
							m_runID_ExitCollimatorSource = m_runID;
							GPUExitCollimatorSource->Fill();
						}
					}
				}

				if( m_launchLastBuffer )
				{
					m_isAlreadyLaunchedBuffer = true;
				}

				// Create new track
        for( unsigned int i = 0; i < sizeAfter; ++i )
        {
						if( m_cpuFlag )
							CreateNewParticle( m_cpuParticle, i );
						else
            	CreateNewParticle( m_gpuParticle, i );
        }

				// Set to zero the number of particles
				if( m_cpuFlag )
        	m_cpuParticle->size = 0;
				else
					m_gpuParticle->size = 0;
    }

	if( nVerboseLevel > 3 )
		G4cout << " >> leaving [GateToGPUImageSPECT::RecordStep]" << G4endl;
}

void GateToGPUImageSPECT::SetVolumeToAttach( G4String& volName )
{
    m_volToAttach = volName;
}

void GateToGPUImageSPECT::CreateNewParticle( GateCPUParticle const *p,
    unsigned int id )
{
		// Rotate the momentum and position
		G4double x = p->px[ id ] * ::cos( m_angle * PI / 180.0 )
			+ p->pz[ id ] * ::sin( m_angle * PI / 180.0 );
		G4double y = p->py[ id ];
		G4double z = p->pz[ id ] * ::cos( m_angle * PI / 180.0 )
			- p->px[ id ] * ::sin( m_angle * PI / 180.0 );

		G4double dx = p->dx[ id ] * ::cos( m_angle * PI / 180.0 )
			+ p->dz[ id ] * ::sin( m_angle * PI / 180.0 );
		G4double dy = p->dy[ id ];
		G4double dz = p->dz[ id ] * ::cos( m_angle * PI / 180.0 )
			- p->dx[ id ] * ::sin( m_angle * PI / 180.0 );

    G4ThreeVector dir( dx, dy, dz );
    dir /= dir.mag();
    G4ThreeVector position( x*mm, y*mm, z*mm);

    G4DynamicParticle *dp = NULL;
    if( p->type[ id ] == 0 )
        dp = new G4DynamicParticle( G4Gamma::Gamma(),
            dir, ( p->E[ id ] )*MeV );
    else if( p->type[ id ] == 1 )
        dp = new G4DynamicParticle( G4Electron::Electron(),
            dir, ( p->E[ id ] )*MeV );

    double time = p->t[ id ];
    G4Track *newTrack = new G4Track( dp, time, position );
    //newTrack->SetParentID( p->parentID[ id ] );
		newTrack->SetParentID( p->eventID[ id ] );
    newTrack->SetTrackID( p->trackID[ id ] );

    static G4EventManager *em = G4EventManager::GetEventManager();
    G4StackManager *sm = em->GetStackManager();
    sm->PushOneTrack( newTrack );
}

void GateToGPUImageSPECT::CreateNewParticle( GateGPUParticle const *p,
    unsigned int id )
{
		// Rotate the momentum and position
		G4double x = p->px[ id ] * ::cos( m_angle * PI / 180.0 )
			+ p->pz[ id ] * ::sin( m_angle * PI / 180.0 );
		G4double y = p->py[ id ];
		G4double z = p->pz[ id ] * ::cos( m_angle * PI / 180.0 )
			- p->px[ id ] * ::sin( m_angle * PI / 180.0 );

		G4double dx = p->dx[ id ] * ::cos( m_angle * PI / 180.0 )
			+ p->dz[ id ] * ::sin( m_angle * PI / 180.0 );
		G4double dy = p->dy[ id ];
		G4double dz = p->dz[ id ] * ::cos( m_angle * PI / 180.0 )
			- p->dx[ id ] * ::sin( m_angle * PI / 180.0 );

    G4ThreeVector dir( dx, dy, dz );
    dir /= dir.mag();
    G4ThreeVector position( x*mm, y*mm, z*mm);

    G4DynamicParticle *dp = NULL;
    if( p->type[ id ] == 0 )
        dp = new G4DynamicParticle( G4Gamma::Gamma(),
            dir, ( p->E[ id ] )*MeV );
    else if( p->type[ id ] == 1 )
        dp = new G4DynamicParticle( G4Electron::Electron(),
            dir, ( p->E[ id ] )*MeV );

    double time = p->t[ id ];
    G4Track *newTrack = new G4Track( dp, time, position );
    //newTrack->SetParentID( p->parentID[ id ] );
		newTrack->SetParentID( p->eventID[ id ] );
    newTrack->SetTrackID( p->trackID[ id ] );

    static G4EventManager *em = G4EventManager::GetEventManager();
    G4StackManager *sm = em->GetStackManager();
    sm->PushOneTrack( newTrack );
}

GateGPUCollimator* GateGPUCollimator_new( int ny_pixel, int nz_pixel,
    G4double septa, G4double fy, G4double fz,
    G4double collimatorHeight, G4double spaceBetweenCollimatorDetector,
    G4double *centerOfPxlY, G4double *centerOfPxlZ, G4double y_pixel_size,
    G4double z_pixel_size, unsigned int cudaDeviceID )
{
    GateGPUCollimator *c = new GateGPUCollimator;
    c->y_size = ny_pixel * 2;
    c->z_size = nz_pixel * 2;
    c->cudaDeviceID = cudaDeviceID;
    c->planeToProject = collimatorHeight - spaceBetweenCollimatorDetector;

    // Computing bounding with septa
    G4double boundSeptaY = ( y_pixel_size - septa ) / 2.0;
    G4double boundSeptaZ = ( z_pixel_size - septa ) / 2.0;

    // Compute entry and exit points in Y
    // Allocate memory for entry and exit
    c->entry_collim_y = new float[ c->y_size ];
    c->exit_collim_y = new float[ c->y_size ];
    // Loop over Y pixel
    G4double boundPlus = 0.0;
    G4double boundMinus = 0.0;
    G4double denominator = 0.0;
    G4double numeratorEntry = 0.0;
    G4double numeratorExit = 0.0;
    for( G4int i = 0, j = 0, k = 0; i < ny_pixel; ++i )
    {
        denominator = fy + collimatorHeight;
        numeratorEntry = fy;
        numeratorExit = fy + collimatorHeight - spaceBetweenCollimatorDetector;
        boundPlus = centerOfPxlY[ i ] + boundSeptaY;
        boundMinus = centerOfPxlY[ i ] - boundSeptaY;
        c->entry_collim_y[ j++ ] = ( numeratorEntry / denominator ) * boundPlus;
        c->entry_collim_y[ j++ ] = ( numeratorEntry / denominator ) * boundMinus;
        c->exit_collim_y[ k++ ] = ( numeratorExit / denominator ) * boundPlus;
        c->exit_collim_y[ k++ ] = ( numeratorExit / denominator ) * boundMinus;
    }

    // Compute entry and exit points in Z
    // Allocate memory for entry and exit
    c->entry_collim_z = new float[ c->z_size ];
    c->exit_collim_z = new float[ c->z_size ];
    for( G4int i = 0, j = 0, k = 0; i < nz_pixel; ++i )
    {
        denominator = fz + collimatorHeight;
        numeratorEntry = fz;
        numeratorExit = fz + collimatorHeight - spaceBetweenCollimatorDetector;
        boundPlus = centerOfPxlZ[ i ] + boundSeptaZ;
        boundMinus = centerOfPxlZ[ i ] - boundSeptaZ;
        c->entry_collim_z[ j++ ] = ( numeratorEntry / denominator ) * boundPlus;
        c->entry_collim_z[ j++ ] = ( numeratorEntry / denominator ) * boundMinus;
        c->exit_collim_z[ k++ ] = ( numeratorExit / denominator ) * boundPlus;
        c->exit_collim_z[ k++ ] = ( numeratorExit / denominator ) * boundMinus;
    }

    return c;
}

void GateGPUCollimator_delete( GateGPUCollimator *in )
{
    if( in )
    {
        delete[] in->entry_collim_y;
        delete[] in->exit_collim_y;
        delete[] in->entry_collim_z;
        delete[] in->exit_collim_z;
        in->y_size = 0;
        in->z_size = 0;
        in = NULL;
    }
}

void GateGPUCollimator_print( GateGPUCollimator* in )
{
    G4cout << "Print collimator informations:" << G4endl;
    if( in )
    {
        G4cout << "Number of pixel elements in Y: "
            << in->y_size/2.0 << G4endl;
        G4cout << "Number of pixel elements in Z: "
            << in->z_size/2.0 << G4endl;
        G4cout << "Number of elements in Y at entry and exit collimator: "
            << in->y_size << G4endl;
        G4cout << "Number of elements in Z at entry and exit collimator: "
            << in->z_size << G4endl;
        G4cout << "position points in Y: " << G4endl;
        for( unsigned int i = 0; i < in->y_size; ++i )
        {
            G4cout << "entry: " << in->entry_collim_y[ i ]
                << "\t\t\t\texit: " << in->exit_collim_y[ i ] << G4endl;
        }
        G4cout << "position points in Z: " << G4endl;
        for( unsigned int i = 0; i < in->z_size; ++i )
        {
            G4cout << "entry: " << in->entry_collim_z[ i ]
                << "\t\t\t\texit: " << in->exit_collim_z[ i ] << G4endl;
        }
    }
}

GateCPUCollimator* GateCPUCollimator_new( int ny_pixel, int nz_pixel,
    G4double septa, G4double fy, G4double fz,
    G4double collimatorHeight, G4double spaceBetweenCollimatorDetector,
    G4double *centerOfPxlY, G4double *centerOfPxlZ, G4double y_pixel_size,
    G4double z_pixel_size )
{
    GateCPUCollimator *c = new GateCPUCollimator;
    c->y_size = ny_pixel * 2;
    c->z_size = nz_pixel * 2;
    c->planeToProject = collimatorHeight - spaceBetweenCollimatorDetector;

    // Computing bounding with septa
    G4double boundSeptaY = ( y_pixel_size - septa ) / 2.0;
    G4double boundSeptaZ = ( z_pixel_size - septa ) / 2.0;

    // Compute entry and exit points in Y
    // Allocate memory for entry and exit
    c->entry_collim_y = new float[ c->y_size ];
    c->exit_collim_y = new float[ c->y_size ];
    // Loop over Y pixel
    G4double boundPlus = 0.0;
    G4double boundMinus = 0.0;
    G4double denominator = 0.0;
    G4double numeratorEntry = 0.0;
    G4double numeratorExit = 0.0;
    for( G4int i = 0, j = 0, k = 0; i < ny_pixel; ++i )
    {
        denominator = fy + collimatorHeight;
        numeratorEntry = fy;
        numeratorExit = fy + collimatorHeight - spaceBetweenCollimatorDetector;
        boundPlus = centerOfPxlY[ i ] + boundSeptaY;
        boundMinus = centerOfPxlY[ i ] - boundSeptaY;
        c->entry_collim_y[ j++ ] = ( numeratorEntry / denominator ) * boundPlus;
        c->entry_collim_y[ j++ ] = ( numeratorEntry / denominator ) * boundMinus;
        c->exit_collim_y[ k++ ] = ( numeratorExit / denominator ) * boundPlus;
        c->exit_collim_y[ k++ ] = ( numeratorExit / denominator ) * boundMinus;
    }

    // Compute entry and exit points in Z
    // Allocate memory for entry and exit
    c->entry_collim_z = new float[ c->z_size ];
    c->exit_collim_z = new float[ c->z_size ];
    for( G4int i = 0, j = 0, k = 0; i < nz_pixel; ++i )
    {
        denominator = fz + collimatorHeight;
        numeratorEntry = fz;
        numeratorExit = fz + collimatorHeight - spaceBetweenCollimatorDetector;
        boundPlus = centerOfPxlZ[ i ] + boundSeptaZ;
        boundMinus = centerOfPxlZ[ i ] - boundSeptaZ;
        c->entry_collim_z[ j++ ] = ( numeratorEntry / denominator ) * boundPlus;
        c->entry_collim_z[ j++ ] = ( numeratorEntry / denominator ) * boundMinus;
        c->exit_collim_z[ k++ ] = ( numeratorExit / denominator ) * boundPlus;
        c->exit_collim_z[ k++ ] = ( numeratorExit / denominator ) * boundMinus;
    }

    return c;
}

void GateCPUCollimator_delete( GateCPUCollimator *in )
{
    if( in )
    {
        delete[] in->entry_collim_y;
        delete[] in->exit_collim_y;
        delete[] in->entry_collim_z;
        delete[] in->exit_collim_z;
        in->y_size = 0;
        in->z_size = 0;
        in = NULL;
    }
}

void GateCPUCollimator_print( GateCPUCollimator* in )
{
    G4cout << "Print collimator informations:" << G4endl;
    if( in )
    {
        G4cout << "Number of pixel elements in Y: "
            << in->y_size/2.0 << G4endl;
        G4cout << "Number of pixel elements in Z: "
            << in->z_size/2.0 << G4endl;
        G4cout << "Number of elements in Y at entry and exit collimator: "
            << in->y_size << G4endl;
        G4cout << "Number of elements in Z at entry and exit collimator: "
            << in->z_size << G4endl;
        G4cout << "position points in Y: " << G4endl;
        for( unsigned int i = 0; i < in->y_size; ++i )
        {
            G4cout << "entry: " << in->entry_collim_y[ i ]
                << "\t\t\t\texit: " << in->exit_collim_y[ i ] << G4endl;
        }
        G4cout << "position points in Z: " << G4endl;
        for( unsigned int i = 0; i < in->z_size; ++i )
        {
            G4cout << "entry: " << in->entry_collim_z[ i ]
                << "\t\t\t\texit: " << in->exit_collim_z[ i ] << G4endl;
        }
    }
}
