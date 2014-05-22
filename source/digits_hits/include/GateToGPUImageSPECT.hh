/*----------------------
   OpenGATE Collaboration

   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \file GateToGPUImageSPECT.hh

  \brief Class GateToGPUImageSPECT
*/

#ifndef GATETOGPUIMAGESPECT_HH
#define GATETOGPUIMAGESPECT_HH

#include "GateVOutputModule.hh"
#include "GateApplicationMgr.hh"
#include "TFile.h"
#include "TTree.h"
#include "GateGPUParticle.hh"
#include "GateCPUParticle.hh"
#include <ctime>

class GateVSystem;
class GateToGPUImageSPECTMessenger;

struct GateGPUCollimator
{
    float          *entry_collim_y;
    float          *entry_collim_z;
    float          *exit_collim_y;
    float          *exit_collim_z;
    unsigned int    y_size;
    unsigned int    z_size;
    unsigned int    cudaDeviceID;
    float          *gpu_entry_collim_y;
    float          *gpu_entry_collim_z;
    float          *gpu_exit_collim_y;
    float          *gpu_exit_collim_z;
    float          planeToProject;
};

struct GateCPUCollimator
{
    float          *entry_collim_y;
    float          *entry_collim_z;
    float          *exit_collim_y;
    float          *exit_collim_z;
    unsigned int    y_size;
    unsigned int    z_size;
    float          planeToProject;
};


typedef struct ThreadArgSPECT_t
{
	GateCPUCollimator   *m_cpuCollimator;
	GateCPUParticle     *m_cpuParticle;
	unsigned int        nThread;
	unsigned int        tID;
} ThreadArgSPECT;

class GateToGPUImageSPECT : public GateVOutputModule
{
	public:
		GateToGPUImageSPECT( const G4String& name, GateOutputMgr *outputMgr,
            GateVSystem* itsSystem, DigiMode digiMode );
		virtual ~GateToGPUImageSPECT();

		virtual void RecordBeginOfAcquisition();
		virtual void RecordEndOfAcquisition();
		virtual void RecordBeginOfRun( const G4Run* );
		virtual void RecordEndOfRun( const G4Run* );
		virtual void RecordBeginOfEvent( const G4Event* );
		virtual void RecordEndOfEvent( const G4Event* );
		virtual void RecordStepWithVolume( const GateVVolume*, const G4Step* );
		virtual void RecordVoxels( GateVGeometryVoxelStore* ){};

        virtual const G4String& GiveNameOfFile();

    public:
        void SetFileName( G4String& );
		inline G4String GetFileName() { return m_fileName; }
        void SetVolumeToAttach( G4String& );
        inline G4String GetVolumeToAttach() { return m_volToAttach; }
        void SetBufferParticleEntry( G4int );
        inline G4int GetBufferParticleEntry() { return m_bufferParticleEntry; }
        void SetCudaDevice( G4int );
        inline G4int GetCudaDevice() { return m_cudaDevice; }
				void SetCpuNumber( G4int );
				void SetCpuFlag( G4bool );
        void CreateNewParticle( GateGPUParticle const* p, unsigned int id );
        void CreateNewParticle( GateCPUParticle const* p, unsigned int id );
        void SetRootHitFlag( G4bool );
        void SetRootSingleFlag( G4bool );
        void SetRootSourceFlag( G4bool );
				void SetRootExitCollimatorSourceFlag( G4bool );
				void SetTimeFlag( G4bool );
        void SetNYpixel( G4int );
        void SetNZpixel( G4int );
				void SetZPixelSize( G4double );
				void SetYPixelSize( G4double );
        void SetSepta( G4double );
        inline G4double GetSepta() { return m_septa; }
        void SetFy( G4double );
        void SetFz( G4double );
        void SetCollimatorHeight( G4double );
        void SetSpaceBetweenCollimatorDetector( G4double );
        void SetRor( G4double );

	protected:
		GateOutputMgr                    *m_outputMgr;
		GateToGPUImageSPECTMessenger     *m_messenger;
        GateVSystem                      *m_system;
        G4String                          m_fileName;
        G4String                          m_volToAttach;
        TFile                            *m_file;
        unsigned int                      m_bufferParticleEntry;
        G4int                             m_cudaDevice;
				G4int                             m_cpuNumber;
				G4bool                            m_cpuFlag;
        G4bool                            m_rootHitFlag;
        G4bool                            m_rootSingleFlag;
        G4bool                            m_rootSourceFlag;
				G4bool                            m_rootExitCollimatorSourceFlag;
				G4bool                            m_timeFlag;
        G4int                             m_ny_pixel;
        G4int                             m_nz_pixel;
				G4double                          m_z_pixel_size;
				G4double                          m_y_pixel_size;

        G4double                         *m_centerOfPxlZ;
        G4double                         *m_centerOfPxlY;
        GateGPUCollimator                *m_gpuCollimator;
        GateCPUCollimator                *m_cpuCollimator;
				G4double                          m_septa;
        G4double                          m_fy;
        G4double                          m_fz;
        G4double                          m_collimatorHeight;
        G4double                          m_spaceBetweenCollimatorDetector;
				G4double                          m_ror;
				G4bool                            m_launchLastBuffer;
				G4bool                            m_isAlreadyLaunchedBuffer;

				G4double                          m_elapsedTime;

        // Input and output structure for gpu.
        GateGPUParticle  *m_gpuParticle;
				GateCPUParticle  *m_cpuParticle;
				G4RotationMatrix  m_matrixRotation;
				G4int m_runID;
				G4double m_angle;

        // Value for ROOT file
        // TTrees
        TTree *GPUHits;
        TTree *GPUSingles;
        TTree *GPUSource;
				TTree *GPUExitCollimatorSource;

        // ROOT values for source
        G4float m_posX_Source, m_posY_Source, m_posZ_Source;
				G4float m_energy_Source;
				G4int m_runID_Source;

				// ROOT values for exit collimator source
				G4float m_posX_ExitCollimatorSource, m_posY_ExitCollimatorSource;
				G4float m_posZ_ExitCollimatorSource;
				G4float m_energy_ExitCollimatorSource;
				G4int m_runID_ExitCollimatorSource;

        // ROOT values for hits
        G4float m_posX_Hit, m_posY_Hit, m_posZ_Hit;
        G4float m_energy_Hit;
        G4float m_time_Hit;
				G4int m_runID_Hit;

        // ROOT values for singles
        G4float m_posX_Single, m_posY_Single, m_posZ_Single;
        G4float m_energy_Single;
        G4float m_time_Single;
				G4float m_src_posX_Single, m_src_posY_Single, m_src_posZ_Single;
				G4int m_runID_Single;
};

// GPU
extern GateGPUCollimator* GateGPUCollimator_new( int ny_pixel, int nz_pixel,
    G4double septa, G4double fy, G4double fz,
    G4double collimatorHeight, G4double spaceBetweenCollimatorDetector,
    G4double *centerOfPxlY, G4double *centerOfPxlZ, G4double y_pixel_size,
    G4double z_pixel_size, unsigned int cudaDeviceID );
extern void GateGPUCollimator_print( GateGPUCollimator *in );
extern G4double GateGPUCollimator_getFocale( G4double z, G4double zMax,
    G4double fzMin, G4double fzMax );
extern void GateGPUCollimator_delete( GateGPUCollimator *in );

// CPU
extern GateCPUCollimator* GateCPUCollimator_new( int ny_pixel, int nz_pixel,
    G4double septa, G4double fy, G4double fz,
    G4double collimatorHeight, G4double spaceBetweenCollimatorDetector,
    G4double *centerOfPxlY, G4double *centerOfPxlZ, G4double y_pixel_size,
    G4double z_pixel_size );
extern void GateCPUCollimator_print( GateCPUCollimator *in );
extern G4double GateCPUCollimator_getFocale( G4double z, G4double zMax,
    G4double fzMin, G4double fzMax );
extern void GateCPUCollimator_delete( GateCPUCollimator *in );

// DEFINED IN GATECOLLIM_GPU.cu
extern void GateGPUCollimator_init( GateGPUCollimator *in );
extern void GateGPUCollimator_process( GateGPUCollimator *in, GateGPUParticle *particle );

#endif
