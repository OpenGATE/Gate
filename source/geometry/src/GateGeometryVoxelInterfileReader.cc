/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <fstream>
/* PY Descourt 08/09/2009 */
#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
/* PY Descourt 08/09/2009 */
#include "GateGeometryVoxelInterfileReader.hh"
#include "GateGeometryVoxelInterfileReaderMessenger.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "GateMaterialDatabase.hh"
#include "GateVVolume.hh"

#include <stdio.h>
#include <string.h>

GateGeometryVoxelInterfileReader::GateGeometryVoxelInterfileReader(GateVVolume* inserter)
  : GateVGeometryVoxelReader(inserter), GateInterfileHeader()
{
  IsFirstFrame = true;
  m_name = G4String("interfileReader");
  m_fileName  = G4String("");
  m_messenger = new GateGeometryVoxelInterfileReaderMessenger(this);
}

GateGeometryVoxelInterfileReader::~GateGeometryVoxelInterfileReader()
{
  if (m_messenger) {
    delete m_messenger;
  }
}


void GateGeometryVoxelInterfileReader::Describe(G4int level)
{
  G4cout << " Voxel reader type ---> " << m_name << G4endl;

  GateVGeometryVoxelReader::Describe(level);

}

void GateGeometryVoxelInterfileReader::ReadFile(G4String headerFileName)
{
  m_fileName = headerFileName;
  ReadHeader(headerFileName);

  std::vector<PixelType> buffer;

  ReadData(buffer);

  EmptyStore();

  G4String materialName;
  G4double   imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;

  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
	imageValue = buffer[ix+nx*iy+nx*ny*iz];
	materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
	if ( materialName != G4String("NULL") ) {
	  G4Material* material = mMaterialDatabase.GetMaterial(materialName);
	  AddVoxel(ix, iy, iz, material);
	} else {
	  G4cout << "GateGeometryVoxelInterfileReader::ReadFile: WARNING: voxel not added (material translation not found); value: "<< imageValue << G4endl;
	}
      }
    }
  }

  UpdateParameters();

  // saves the voxel info through the OutputMgr
  Dump();

  if (m_compressor) {
    Compress();
    EmptyStore();
    G4cout << "GateSourceVoxelInterfileReader::ReadFile: For your information, the voxel store has been emptied." << G4endl;
  }
}

/*PY Descourt 08/09/2009 */
void GateGeometryVoxelInterfileReader::ReadRTFile(G4String headerFileName, G4String dataFileName)
{
  // Check if there is a GatePhantom attached to this source
  GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckGeometryAttached( GetCreator()->GetObjectName() );

  if ( Ph != 0) {
      G4cout << " The Object "<< Ph->GetName()
		<<" is attached to the "<<m_name<<" Geometry Voxel Reader."<<G4endl;
  } else {
      G4cout << " GateGeometryVoxelInterfileReader::ReadFile   WARNING The Object "<< Ph->GetName()
	    <<" is not attached to any Geometry Voxel Reader."<<G4endl;
  }

  if ( IsFirstFrame == true ) {
      ReadHeader(headerFileName);
      IsFirstFrame = false;
  }

  // override filename from header
  m_dataFileName = dataFileName;

  std::vector<PixelType> buffer;

  ReadData(m_dataFileName, buffer);

  EmptyStore();

  G4String materialName;
  G4int    imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;
  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;

  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
      for (G4int iy=0; iy<ny; iy++) {
	  for (G4int ix=0; ix<nx; ix++) {
	      imageValue = buffer[ix+nx*iy+nx*ny*iz];
	      materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
	      if ( materialName != G4String("NULL") ) {
		  G4Material* material = mMaterialDatabase.GetMaterial(materialName);
		  AddVoxel(ix, iy, iz, material);
	      } else {
		  G4cout << "GateGeometryVoxelInterfileReader::ReadFile: WARNING: voxel not added (material translation not found); value: "<< imageValue << G4endl;
	      }
	  }
      }
  }

  if (m_compressor) {
      m_compressor->Initialize();
      Compress();

      G4cout << "---------- Gate Voxels Compressor Statistics ---------"<<G4endl;
      G4cout << "  Initial number of voxels in The Phantom      : " << GetNumberOfVoxels() << G4endl;
      G4cout << "  number of compressed voxels                  : " << m_compressor->GetNbOfCopies() << G4endl;
      G4cout << "  Compression achieved                                            : " << m_compressor->GetCompressionRatio() << " %"  << G4endl;
      G4cout << "-------------------------------------------------------------------"<<G4endl;
  }
}
