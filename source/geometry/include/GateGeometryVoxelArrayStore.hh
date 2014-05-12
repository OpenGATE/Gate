/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateGeometryVoxelArrayStore_h
#define GateGeometryVoxelArrayStore_h 1

#include "GateVGeometryVoxelStore.hh"
#include "GateVoxelCompressor.hh"

/*! \class  GateGeometryVoxelArrayStore
    \brief  This class can read a file with the material info for a matrix of voxels (digital image)
    
    - GateGeometryVoxelArrayStore - by Giovanni.Santin@cern.ch
    
    - It reads the image file, asks the translator to convert the digital info into material info, and 
      stores the material information. This material info is used by the GateVoxelReplicaMatrixInserter
      at the moment of the matrix construction

    - The translator has to be inserted befor reading the image file

    - The material information is a 3D array, to speed-up the computation
      
      \sa GateGeometryVoxelArrayStoreMessenger
      \sa GateVGeometryVoxelTranslator
      \sa GateVSourceVoxelReader
      \sa GateVSourceVoxelTranslator
*/      

class GateGeometryVoxelArrayStore : public GateVGeometryVoxelStore
{

  friend class GateVoxelCompressor;

public:

  GateGeometryVoxelArrayStore(GateVVolume* inserter);
  virtual ~GateGeometryVoxelArrayStore();

  //! it adds an entry in the voxel store
  virtual void             AddVoxel(G4int ix, G4int iy, G4int iz, G4Material* material);

  //! it sets the material of the voxel in the logical position (ix,iy,iz) 
  virtual void             SetVoxelMaterial(G4int ix, G4int iy, G4int iz, G4Material* material);

  //! it gives the material of the voxel in the logical position (ix,iy,iz) 
  virtual G4Material*      GetVoxelMaterial(G4int ix, G4int iy, G4int iz);
  //! it gives the material of the voxel in the logical position (ix,iy,iz) NO CHECK of indices!!
  virtual inline G4Material*      GetVoxelMaterial_noCheck(G4int ix, G4int iy, G4int iz) const;

  //! useful when the voxels have a non cubic distribution: in a loop, with this first you get the 3 indices, then with the 3 indices you ask the material
  virtual std::vector<G4int> GetVoxel(G4int index);

  //! give the number of the stored voxels
  virtual G4int            GetNumberOfVoxels()               { return m_voxelNx * m_voxelNy * m_voxelNz; };

  virtual void             SetVoxelNx(G4int n);
  virtual void             SetVoxelNy(G4int n);
  virtual void             SetVoxelNz(G4int n);

  virtual void             Describe(G4int level);


  void CreateCompressor();
  void Compress();
  const GateVoxelCompressor& GetCompressor() const { return *m_compressor;}
  GateVoxelCompressor* GetCompressorPtr() { return m_compressor;};


protected:

  typedef G4Material**     GateVoxelMaterialArray;
  GateVoxelMaterialArray   m_geometryVoxelMaterials;

  GateVoxelCompressor*     m_compressor;

protected:

  inline G4int             RealArrayIndex(G4int ix, G4int iy, G4int iz, G4int nx, G4int ny, G4int nz) 
  { 
    G4int rai =  ix + iy*nx + iz*nx*ny; 
    //    G4cout << " " << rai;
    if (rai<0 || rai>=nx*ny*nz) G4cout << "RealArrayIndex: ERROR: index out of range!!!!" << G4endl;
    return rai;
  };
  inline G4int             RealArrayIndex_noCheck(G4int ix, G4int iy, G4int iz) const;

  //! updates the sizes nX/nY/nZ. used after reading the file
  virtual void             UpdateParameters();

  //! it deletes all the voxels in the stores, e.g. before reading a new file
  virtual void             EmptyStore();

  //! it initializes all the voxels in the stores
  virtual void             InitStore(GateVoxelMaterialArray store, G4int nx, G4int ny, G4int nz);
};

inline G4int             GateGeometryVoxelArrayStore::RealArrayIndex_noCheck(G4int ix, G4int iy, G4int iz) const
{
	return ix + iy*m_voxelNx + iz*m_voxelNx*m_voxelNy;
}
inline G4Material* GateGeometryVoxelArrayStore::GetVoxelMaterial_noCheck(G4int ix, G4int iy, G4int iz) const
{
	return   m_geometryVoxelMaterials[RealArrayIndex_noCheck(ix,iy,iz)];
}

#endif
