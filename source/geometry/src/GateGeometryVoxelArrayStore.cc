/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateGeometryVoxelArrayStore.hh"
#include "G4Material.hh"


GateGeometryVoxelArrayStore::GateGeometryVoxelArrayStore(GateVVolume* inserter)
  : GateVGeometryVoxelStore(inserter)
    , m_geometryVoxelMaterials(0),m_compressor(0)
{
}

GateGeometryVoxelArrayStore::~GateGeometryVoxelArrayStore()
{
  if (m_compressor) delete m_compressor;
}

void GateGeometryVoxelArrayStore::Describe(G4int level) 
{

  GateVGeometryVoxelStore::Describe(level);

  G4cout << "  number of voxels       : " << GetNumberOfVoxels() << G4endl;
  if (m_compressor)
    G4cout << "  Compression achieved   : " << m_compressor->GetCompressionRatio() << " %"  << G4endl;

  if (level > 2) {
    if (m_geometryVoxelMaterials != 0) {
      for (G4int iz=0; iz<m_voxelNz; iz++) {
	for (G4int iy=0; iy<m_voxelNy; iy++) {
	  for (G4int ix=0; ix<m_voxelNx; ix++) {
	    G4cout << "   Index" 
		   << " " << ix
		   << " " << iy
		   << " " << iz
		   << " Material " << (m_geometryVoxelMaterials[RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz)])->GetName() << G4endl;
	  }
	}
      }
    }
  }
}

void GateGeometryVoxelArrayStore::EmptyStore()
{
  if (m_geometryVoxelMaterials) {
    delete[] m_geometryVoxelMaterials;
    m_geometryVoxelMaterials = 0;
    // m_voxelNx = m_voxelNy = m_voxelNz = 0; BY RTA JUST TO SEE IF IT'S WORTH IT
  }
}

void GateGeometryVoxelArrayStore::InitStore(GateVoxelMaterialArray store, G4int nx, G4int ny, G4int nz)
{
  if (store) {
    for (G4int i=0; i<nx*ny*nz; i++) {
      store[i] = m_defaultMaterial;
    }    
  }
}

void GateGeometryVoxelArrayStore::AddVoxel(G4int ix, G4int iy, G4int iz, G4Material* material)
{
  // this method is used by the ReadFile method of the VoxelReaders. 
  if (m_geometryVoxelMaterials) {
    if ((ix<m_voxelNx) && (iy<m_voxelNy) && (iz<m_voxelNz)) {
      m_geometryVoxelMaterials[RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz)] = material;
    }
  }
}

void GateGeometryVoxelArrayStore::SetVoxelMaterial(G4int ix, G4int iy, G4int iz, G4Material* material)
{
  AddVoxel(ix, iy, iz, material);
}

G4Material* GateGeometryVoxelArrayStore::GetVoxelMaterial(G4int ix, G4int iy, G4int iz)
{
  // this method is used by the SetMaterial method of the ReplicaMatrixInserter.
  G4Material* material = m_defaultMaterial;

  if (m_geometryVoxelMaterials) {
    if ((ix<m_voxelNx) && (iy<m_voxelNy) && (iz<m_voxelNz)) {
      material = m_geometryVoxelMaterials[RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz)];
    } else {
      G4cout << "GateGeometryVoxelArrayStore::GetVoxelMaterial: WARNING: requested voxel position outside the present voxel store" << G4endl;
    } 
  } else {
    G4cout << "GateGeometryVoxelArrayStore::GetVoxelMaterial: WARNING: voxel store not yet defined " << G4endl;
  }

  return material;
}

std::vector<G4int> GateGeometryVoxelArrayStore::GetVoxel(G4int index)
{
  std::vector<G4int> indexVec;
  G4int ix, iy, iz;

  if (index < GetNumberOfVoxels()) {
    iz = index / m_voxelNz;
    iy = (GetNumberOfVoxels() - iz*m_voxelNz) / m_voxelNy;
    ix = (GetNumberOfVoxels() - iz*m_voxelNz) - iy * m_voxelNy;
    indexVec.push_back(ix);
    indexVec.push_back(iy);
    indexVec.push_back(iz);
  }
  return indexVec;
}

void GateGeometryVoxelArrayStore::SetVoxelNx(G4int n)
{  
  if (n>0) {

    G4int nMax = n;
    if (m_voxelNx < n) nMax = m_voxelNx;

    if ((m_voxelNy>0)&&(m_voxelNz>0)) {

      GateVoxelMaterialArray newStore = new G4Material*[n * m_voxelNy * m_voxelNz];
      InitStore(newStore,n,m_voxelNy,m_voxelNz);

      if (m_geometryVoxelMaterials) {

	for (G4int iz=0; iz<m_voxelNz; iz++) {
	  for (G4int iy=0; iy<m_voxelNy; iy++) {
	    for (G4int ix=0; ix<nMax; ix++) {
//  	      G4cout << "GateGeometryVoxelArrayStore::SetVoxelNx " 
//  		     << " Nxyz " 
//  		     << " " << m_voxelNx
//  		     << " " << m_voxelNy
//  		     << " " << m_voxelNz
//  		     << " Index" 
//  		     << " " << ix
//  		     << " " << iy
//  		     << " " << iz
//  		     << " RealArrayIndex old " 
//  		     << RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz) 
//  		     << " new " 
//  		     << RealArrayIndex(ix,iy,iz,        n,m_voxelNy,m_voxelNz) 
//  		     << G4endl;
	      newStore[ RealArrayIndex(ix,iy,iz,n,m_voxelNy,m_voxelNz) ] = 
		m_geometryVoxelMaterials[ RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz) ];
	    }
	  }
	}
	delete[] m_geometryVoxelMaterials;
      }
      m_geometryVoxelMaterials = newStore;
    }
    m_voxelNx = n;
  }
}

void GateGeometryVoxelArrayStore::SetVoxelNy(G4int n)
{
  if (n>0) {

    G4int nMax = n;
    if (m_voxelNy < n) nMax = m_voxelNy;

    if ((m_voxelNx>0)&&(m_voxelNz>0)) {

      GateVoxelMaterialArray newStore = new G4Material*[m_voxelNx * n * m_voxelNz];
      InitStore(newStore,m_voxelNx,n,m_voxelNz);

      if (m_geometryVoxelMaterials) {

	for (G4int iz=0; iz<m_voxelNz; iz++) {
	  for (G4int iy=0; iy<nMax; iy++) {
	    for (G4int ix=0; ix<m_voxelNx; ix++) {
	      newStore[ RealArrayIndex(ix,iy,iz,m_voxelNx,n,m_voxelNz) ] = 
		m_geometryVoxelMaterials[ RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz) ];
	    }
	  }
	}
	delete[] m_geometryVoxelMaterials;
      }
      m_geometryVoxelMaterials = newStore;
    }
    m_voxelNy = n;
  }
}

void GateGeometryVoxelArrayStore::SetVoxelNz(G4int n)
{
  if (n>0) {

    G4int nMax = n;
    if (m_voxelNz < n) nMax = m_voxelNz;

    if ((m_voxelNx>0)&&(m_voxelNy>0)) {

      GateVoxelMaterialArray newStore = new G4Material*[m_voxelNx * m_voxelNy * n];
      InitStore(newStore,m_voxelNx,m_voxelNy,n);

      if (m_geometryVoxelMaterials) {

	for (G4int iz=0; iz<nMax; iz++) {
	  for (G4int iy=0; iy<m_voxelNy; iy++) {
	    for (G4int ix=0; ix<m_voxelNx; ix++) {
	      newStore[ RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,n) ] = 
		m_geometryVoxelMaterials[ RealArrayIndex(ix,iy,iz,m_voxelNx,m_voxelNy,m_voxelNz) ];
	    }
	  }
	}
	delete[] m_geometryVoxelMaterials;
      }
      m_geometryVoxelMaterials = newStore;
    }
    m_voxelNz = n;
  }
}

void GateGeometryVoxelArrayStore::UpdateParameters()
{
}

void GateGeometryVoxelArrayStore::CreateCompressor()
{
  m_compressor = new GateVoxelCompressor(this);
}

void GateGeometryVoxelArrayStore::Compress()
{
  m_compressor->Compress();
}
