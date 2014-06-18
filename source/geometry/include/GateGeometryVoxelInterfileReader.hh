/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelInterfileReader_h
#define GateGeometryVoxelInterfileReader_h 1
#include "GateVVolume.hh"

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <map>

#include "GateVGeometryVoxelReader.hh"
#include "GateInterfileHeader.hh"

class GateGeometryVoxelInterfileReaderMessenger;

/*! \class  GateGeometryVoxelInterfileReader
    \brief  This class is a concrete implementation of the abstract GateVGeometryVoxelReader

    - GateGeometryVoxelInterfileReader - by Giovanni.Santin@cern.ch

    - it reads a the voxel info from an ASCII file with the sequence of info: 
      + nx ny nz
      + dx dy dz (mm)
      + number_of_listed_voxels
      + i1 j1 k1 materialName1
      + i2 j2 k2 materialName2
      + ...

      \sa GateVGeometryVoxelReader
      \sa GateGeometryVoxelInterfileReaderMessenger
      \sa GateVGeometryVoxelTranslator
      \sa GateVSourceVoxelReader
      \sa GateVSourceVoxelTranslator
*/      

class GateGeometryVoxelInterfileReader : public GateVGeometryVoxelReader, public GateInterfileHeader
{
public:
  GateGeometryVoxelInterfileReader(GateVVolume* inserter);
  
  virtual ~GateGeometryVoxelInterfileReader();

  virtual void Describe(G4int level);

  virtual void ReadFile(G4String fileName);

  /*PY Descourt 08/09/2009 */
  virtual void ReadRTFile(G4String header_fileName, G4String fileName);
  /*PY Descourt 08/09/2009 */
  
protected:
  GateGeometryVoxelInterfileReaderMessenger* m_messenger;
  G4bool IsFirstFrame; // for RTPhantom
};

#endif


