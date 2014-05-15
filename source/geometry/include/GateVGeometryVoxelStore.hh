/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVGeometryVoxelStore_h
#define GateVGeometryVoxelStore_h 1

#include "GateVVolume.hh"

#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <map>


class G4Material;

/*! \class  GateVGeometryVoxelStore
    \brief  This class can store the material info for a matrix of voxels (digital image)
    
    - GateVGeometryVoxelStore - by Giovanni.Santin@cern.ch
    
    - It stores the material information for the image voxels and allows to access the info through 
      several methods, for random or sequential access. 

    - The material info can be filled in several ways, e.g. read by a file reader  

    - The voxel material store is not concrete in this class, and can be of different types, e.g. 
      a simple 3D matrix (3D array) or a generic map to allow a generic number of spread voxels, for 
      example for non-cubic voxel material distribution (like voxels in a spheric envelope). 
      (The power of the generic map is exploited only by a geometry
      that allows a voxel placement different from a complete 3D matrix.)
      
      \sa GateVGeometryVoxelStoreMessenger
      \sa GateVGeometryVoxelTranslator
      \sa GateVSourceVoxelReader
      \sa GateVSourceVoxelTranslator
*/      

class GateVGeometryVoxelStore
{
public:

  GateVGeometryVoxelStore(GateVVolume* creator);
  virtual ~GateVGeometryVoxelStore();

//E  GateVVolume*     GetInserter()  { return m_inserter; };
  GateVVolume*     GetCreator()  { return m_creator; };
  
  G4String                 GetName()      { return m_name; };

  virtual void             SetVerboseLevel(G4int value) { nVerboseLevel = value; };

  //! it deletes all the voxels in the stores, e.g. before reading a new file
  virtual void             EmptyStore() = 0;

  //! it adds an entry in the voxel store
  virtual void             AddVoxel(G4int ix, G4int iy, G4int iz, G4Material* material) = 0;

  //! it sets the material of the voxel in the logical position (ix,iy,iz) 
  virtual void             SetVoxelMaterial(G4int ix, G4int iy, G4int iz, G4Material* material) = 0;

  //! it gives the material of the voxel in the logical position (ix,iy,iz) 
  virtual G4Material*      GetVoxelMaterial(G4int ix, G4int iy, G4int iz) = 0;

  //! useful when the voxels have a non cubic distribution: in a loop, with this first you get the 3 indices, then with the 3 indices you ask the material
  virtual std::vector<G4int> GetVoxel(G4int index) = 0;

  virtual G4int            GetNumberOfVoxels() = 0;

  virtual G4int            GetVoxelNx()        { return m_voxelNx; };
  virtual void             SetVoxelNx(G4int n) { m_voxelNx = n; };

  virtual G4int            GetVoxelNy()        { return m_voxelNy; };
  virtual void             SetVoxelNy(G4int n) { m_voxelNy = n; };

  virtual G4int            GetVoxelNz()        { return m_voxelNz; };
  virtual void             SetVoxelNz(G4int n) { m_voxelNz = n; };

  virtual void             SetVoxelSize(G4ThreeVector size)  { m_voxelSize = size; };
  virtual G4ThreeVector    GetVoxelSize()                    { return m_voxelSize; };

  virtual void             SetPosition(G4ThreeVector pos)    { m_position = pos; };
  virtual G4ThreeVector    GetPosition()                     { return m_position; };

  virtual void             SetDefaultMaterial(G4Material* material)  { m_defaultMaterial = material; };
  virtual void             SetDefaultMaterial(G4String materialName);
  virtual G4Material*      GetDefaultMaterial()                      { return m_defaultMaterial; };

  virtual void             Dump();

  virtual void             Describe(G4int level);

protected:

  G4int                          nVerboseLevel;

  G4String                       m_name;

//E  GateVVolume*           m_inserter;

  GateVVolume*                   m_creator;
  
  G4int                          m_voxelNx;
  G4int                          m_voxelNy;
  G4int                          m_voxelNz;

  G4ThreeVector                  m_voxelSize;

  G4ThreeVector                  m_position;

  G4Material*                    m_defaultMaterial;

  G4String                       m_type;

};

#endif


