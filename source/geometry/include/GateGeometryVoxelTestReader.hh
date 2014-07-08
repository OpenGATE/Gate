/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelTestReader_h
#define GateGeometryVoxelTestReader_h 1

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <map>

#include "GateVGeometryVoxelReader.hh"

class GateGeometryVoxelTestReaderMessenger;

/*! \class  GateGeometryVoxelTestReader
    \brief  This class is a concrete implementation of the abstract GateVGeometryVoxelReader

    - GateGeometryVoxelTestReader - by Giovanni.Santin@cern.ch

    - it reads a the voxel info from an ASCII file with the sequence of info: 
      + nx ny nz
      + dx dy dz (mm)
      + number_of_listed_voxels
      + i1 j1 k1 materialName1
      + i2 j2 k2 materialName2
      + ...

      \sa GateVGeometryVoxelReader
      \sa GateGeometryVoxelTestReaderMessenger
      \sa GateVGeometryVoxelTranslator
      \sa GateVSourceVoxelReader
      \sa GateVSourceVoxelTranslator
*/      

class GateGeometryVoxelTestReader : public GateVGeometryVoxelReader
{
public:

  GateGeometryVoxelTestReader(GateVVolume* inserter);
  virtual ~GateGeometryVoxelTestReader();

  virtual void ReadFile(G4String fileName);

  virtual void Describe(G4int level);

protected:

  GateGeometryVoxelTestReaderMessenger* m_messenger; 
};

#endif


