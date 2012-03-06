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

  virtual void Update();

  virtual G4int GeneratePrimaries(G4Event* event);

  void ReaderInsert(G4String readerType);

  void ReaderRemove();


protected:

  GateSourceGPUVoxellizedMessenger* m_sourceVoxellizedMessenger;

  // Even if for a standard source the position is completely controlled by its GPS,
  // for Voxel sources (and maybe in the future for all sources) a position and a 
  // rotation are needed to align them to a Geometry Voxel Matrix

  G4ThreeVector                  m_sourcePosition;
  G4RotationMatrix               m_sourceRotation;

  GateVSourceVoxelReader*        m_voxelReader;

  GateSourceGPUVoxellizedInput * m_gpu_input;
  GateSourceGPUVoxellizedOutputParticles * m_gpu_output;
  
};

#endif
