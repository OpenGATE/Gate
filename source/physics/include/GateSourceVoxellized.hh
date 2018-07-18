/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATESOURCEVOXELLIZED_H
#define GATESOURCEVOXELLIZED_H 1

#include "globals.hh"
#include "GateVSource.hh"

class GateSourceVoxellizedMessenger;
class GateVSourceVoxelReader;

//-----------------------------------------------------------------------------
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

  GateVSourceVoxelReader* GetReader() { return m_voxelReader; }

  void          SetPosition(G4ThreeVector pos) { m_sourcePosition = pos; };

  G4ThreeVector GetPosition()                  { return m_sourcePosition; };

  void          SetIsoCenterPosition(G4ThreeVector pos);

protected:

  GateSourceVoxellizedMessenger* m_sourceVoxellizedMessenger;

  // Even if for a standard source the position is controlled by its GPS,
  // for Voxel sources (and maybe in the future for all sources) a position and a
  // rotation are needed to align them to a Geometry Voxel Matrix

  G4ThreeVector                  m_sourcePosition;
  G4RotationMatrix               m_sourceRotation;
  GateVSourceVoxelReader*        m_voxelReader;
};
//-----------------------------------------------------------------------------

#endif
