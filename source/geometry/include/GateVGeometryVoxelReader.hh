/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVGeometryVoxelReader_h
#define GateVGeometryVoxelReader_h 1

#include "GateVVolume.hh"
#include "GateMaterialDatabase.hh"

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <map>

class GateVGeometryVoxelTranslator;


//#include "GateGeometryVoxelMapStore.hh"
#include "GateGeometryVoxelArrayStore.hh"
#include "G4VisAttributes.hh"

/*! \class  GateVGeometryVoxelReader
    \brief  This class can read a file with the material info for a matrix of voxels (digital image)
    
    - GateVGeometryVoxelReader - by Giovanni.Santin@cern.ch
    
    - It reads the image file, asks the translator to convert the digital info into material info, and 
      stores the material information. This material info is used by the GateVoxelReplicaMatrixInserter
      at the moment of the matrix construction

    - The translator has to be inserted befor reading the image file

    - The material information is a generic map to allow for non cubic voxel material distribution 
      (like voxels in a spheric envelope)
      
      \sa GateVGeometryVoxelReaderMessenger
      \sa GateVGeometryVoxelTranslator
      \sa GateVSourceVoxelReader
      \sa GateVSourceVoxelTranslator
*/      

//class GateVGeometryVoxelReader : public GateGeometryVoxelMapStore
class GateVGeometryVoxelReader : public GateGeometryVoxelArrayStore
{
public:

  GateVGeometryVoxelReader(GateVVolume* inserter);
  virtual ~GateVGeometryVoxelReader();

  virtual void             InsertTranslator(G4String translatorType);
  virtual void             RemoveTranslator();

  virtual void             ReadFile(G4String fileName) = 0;
  virtual void             ReadRTFile(G4String header_fileName, G4String fileName) = 0; /* PY Descourt 08/09/2009 */
  virtual void             Describe(G4int level);

  virtual GateVGeometryVoxelTranslator*  GetVoxelTranslator() { return m_voxelTranslator; };

  virtual inline G4Material*      GetVoxelMaterial(G4int ix, G4int iy, G4int iz){
    return GateGeometryVoxelArrayStore::GetVoxelMaterial(ix, iy, iz);
  }

  // This is inlined because it is called by the parameterization and has to be fast
  virtual inline  G4Material*      GetVoxelMaterial(G4int copyNo){
    if (m_geometryVoxelMaterials) return m_geometryVoxelMaterials[copyNo];
    else G4Exception( "GateGeometryVoxelImageReader::GetVoxelMaterial", "GetVoxelMaterial", FatalException, "No material array" );
    return m_geometryVoxelMaterials[copyNo];   //  not really, just to silence the warning
  }

protected:

  GateVGeometryVoxelTranslator*  m_voxelTranslator;

  G4String                       m_fileName;

  GateMaterialDatabase          mMaterialDatabase;
};

#endif
