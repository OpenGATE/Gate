/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceVoxellized_h
#define GateSourceVoxellized_h 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include <vector>
#include <map>
#include "GateVSource.hh"

class GateSourceVoxellizedMessenger;
class GateVSourceVoxelReader;

class GateSourceVoxellized : public GateVSource
{
public:

  GateSourceVoxellized(G4String name);
  virtual ~GateSourceVoxellized();

  virtual G4double GetNextTime(G4double timeNow);

  virtual void Dump(G4int level);

  virtual void Update(G4double time);

  virtual G4int GeneratePrimaries(G4Event* event);

  void ReaderInsert(G4String readerType);

  void ReaderRemove();
  GateVSourceVoxelReader* GetReader() { return m_voxelReader;};/* PY Descourt 08/09/2009 */
  void          SetPosition(G4ThreeVector pos) { m_sourcePosition = pos; };
  G4ThreeVector GetPosition()                  { return m_sourcePosition; };

protected:

  GateSourceVoxellizedMessenger* m_sourceVoxellizedMessenger;

  // Even if for a standard source the position is completely controlled by its GPS,
  // for Voxel sources (and maybe in the future for all sources) a position and a 
  // rotation are needed to align them to a Geometry Voxel Matrix

  G4ThreeVector                  m_sourcePosition;
  G4RotationMatrix               m_sourceRotation;

  GateVSourceVoxelReader*        m_voxelReader;
};

#endif
