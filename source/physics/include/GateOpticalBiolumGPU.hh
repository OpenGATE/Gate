/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEOPTICALBIOLUMGPU_H
#define GATEOPTICALBIOLUMGPU_H 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include <vector>
#include <map>
#include "GateVSource.hh"
#include "GateSourceVoxellized.hh"
#include "GateGPUIO.hh"

class GateOpticalBiolumGPUMessenger;
class GateVSourceVoxelReader;

class GateOpticalBiolumGPU : public GateSourceVoxellized
{
public:

  GateOpticalBiolumGPU(G4String name);
  virtual ~GateOpticalBiolumGPU();

  //virtual G4double GetNextTime(G4double timeNow);

  virtual void Dump(G4int level);

  virtual void Update(double time);

  virtual G4int GeneratePrimaries(G4Event* event);

  virtual void ReaderInsert(G4String readerType);

  virtual void ReaderRemove();

  void AttachToVolume(const G4String& volume_name);

  void SetGPUBufferSize(int n);

  void SetGPUDeviceID(int n);

// vesna
  void SetGPUOpticalPhotonEnergy(double energy);
// vesna

protected:

  GateOpticalBiolumGPUMessenger* m_opticalBiolumGPUMessenger;

  GateGPUIO_Input  * m_gpu_input;
  GateGPUIO_Output * m_gpu_output;
  unsigned int m_current_particle_index_in_buffer;

  // Used for creating a primary
  G4ParticleDefinition* opticalphoton_particle_definition;
  
  // Name of the attached volume
  G4String attachedVolumeName;
  
  // FIXME
  int mNumberOfNextTime;
  int mCurrentTimeID;

  int mCudaDeviceID;

  void GeneratePrimaryEventFromGPUOutput(const GateGPUIO_Particle & particle, G4Event * event);  
  void SetPhantomVolumeData();
};

#endif
