/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATESOURCEGPUVOXELLIZED_H
#define GATESOURCEGPUVOXELLIZED_H 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include <vector>
#include <map>
#include "GateVSource.hh"
#include "GateSourceVoxellized.hh"
#include "GateSourceGPUVoxellizedIO.hh"

class GateSourceGPUVoxellizedMessenger;
class GateVSourceVoxelReader;

class GateSourceGPUVoxellized : public GateSourceVoxellized
{
public:

  GateSourceGPUVoxellized(G4String name);
  virtual ~GateSourceGPUVoxellized();

  virtual G4double GetNextTime(G4double timeNow);

  virtual void Dump(G4int level);

  virtual void Update(double time);

  virtual G4int GeneratePrimaries(G4Event* event);

  virtual void ReaderInsert(G4String readerType);

  virtual void ReaderRemove();

  void AttachToVolume(const G4String& volume_name);


protected:

  GateSourceGPUVoxellizedMessenger* m_sourceGPUVoxellizedMessenger;

  GateSourceGPUVoxellizedInput * m_gpu_input;
  GateSourceGPUVoxellizedOutput  m_gpu_output;

  // Used for creating a primary
  G4ParticleDefinition* gamma_particle_definition;
  
  // Name of the attached volume
  G4String attachedVolumeName;

  void GeneratePrimaryEventFromGPUOutput(GateSourceGPUVoxellizedOutputParticle & particle, G4Event * event);  
  void SetPhantomVolumeData();
};

#endif
