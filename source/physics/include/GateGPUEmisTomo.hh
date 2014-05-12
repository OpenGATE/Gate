/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGPUEMISTOMO_H
#define GATEGPUEMISTOMO_H 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include <vector>
#include <map>
#include "GateVSource.hh"
#include "GateSourceVoxellized.hh"
#include "GateGPUIO.hh"
#include "GateApplicationMgr.hh"

class GateGPUEmisTomoMessenger;
class GateVSourceVoxelReader;

class GateGPUEmisTomo : public GateSourceVoxellized
{
public:

  GateGPUEmisTomo(G4String name);
  virtual ~GateGPUEmisTomo();

  //virtual G4double GetNextTime(G4double timeNow);

  virtual void Dump(G4int level);

  virtual void Update(double time);

  virtual G4int GeneratePrimaries(G4Event* event);

  virtual void ReaderInsert(G4String readerType);

  virtual void ReaderRemove();

  void AttachToVolume(const G4String& volume_name);

  void SetGPUBufferSize(int n);

  void SetGPUDeviceID(int n);

protected:

  GateGPUEmisTomoMessenger* m_sourceGPUVoxellizedMessenger;

  GateGPUIO_Input  * m_gpu_input;
  GateGPUIO_Output * m_gpu_output;

  // Used for creating a primary
  G4ParticleDefinition* gamma_particle_definition;
  
  // Name of the attached volume
  G4String attachedVolumeName;
  
  // Init GPU
  Materials gpu_materials;
  Volume gpu_phantom;
  Activities gpu_activities;
  StackParticle gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2;


  // Half phantom size
  float half_phan_size_x, half_phan_size_y, half_phan_size_z;

  // Stack management
  unsigned int max_buffer_size;
  unsigned int nb_event_in_buffer;
  unsigned int id_event_in_buffer;
  unsigned int id_event;
  double current_time;

  int tot_p;

  // FIXME
  //int mNumberOfNextTime;

  int mUserCount;

  int mCudaDeviceID;

  int mBeginRunFlag;

  //void GeneratePrimaryEventFromGPUOutput(const GateGPUIO_Particle & particle, G4Event * event);  
  void SetPhantomVolumeData();
};

#endif
