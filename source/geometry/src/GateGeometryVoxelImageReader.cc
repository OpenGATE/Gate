/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <fstream>
#include "GateRTPhantom.hh" /*PY Descourt 11/12/2008 */
#include "GateRTPhantomMgr.hh" /*PY Descourt 11/12/2008 */
#include "GateGeometryVoxelImageReader.hh"
#include "GateGeometryVoxelImageReaderMessenger.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "GateMaterialDatabase.hh"
#include "GateVVolume.hh"

GateGeometryVoxelImageReader::GateGeometryVoxelImageReader(GateVVolume* inserter)
  : GateVGeometryVoxelReader(inserter)
{
  m_name = G4String("imageReader");
  m_fileName  = G4String("");
  m_messenger = new GateGeometryVoxelImageReaderMessenger(this);
}

GateGeometryVoxelImageReader::~GateGeometryVoxelImageReader()
{
  if (m_messenger) {
    delete m_messenger;
  }
}


void GateGeometryVoxelImageReader::Describe(G4int level) 
{
  G4cout << " Voxel reader type ---> " << m_name << G4endl;

  GateVGeometryVoxelReader::Describe(level);
  
}

void GateGeometryVoxelImageReader::ReadFile(G4String fileName)
{
  if (m_voxelTranslator == NULL) {
    G4cout << "GateGeometryVoxelImageReader::ReadFile: WARNING: Insert the translator before reading the image" << G4endl
	   << "                                                 Reading aborted." << G4endl;
    return;
  }

  m_fileName = fileName;

  EmptyStore();

  std::ifstream inFile;
  G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4String materialName; 
  G4int    imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  inFile >> nx >> ny >> nz;
  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;
  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  inFile >> dx >> dy >> dz;
  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;
  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
	inFile >> imageValue;
	materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
	if ( materialName != G4String("NULL") ) {
	  G4Material* material = mMaterialDatabase.GetMaterial(materialName);
	  AddVoxel(ix, iy, iz, material);
	} else {
	  G4cout << "GateSourceVoxelImageReader::ReadFile: WARNING: voxel not added (material translation not found)" << G4endl;
	}
      }
    }
  }

  inFile.close();

  UpdateParameters();

  // saves the voxel info through the OutputMgr
  Dump();

  if (m_compressor) {
    Compress();
    EmptyStore();
    G4cout << "GateSourceVoxelImageReader::ReadFile: For your information, the voxel store has been emptied." << G4endl;
  }
  

}

/*PY Descourt 08/09/2009 */

void GateGeometryVoxelImageReader::ReadRTFile(G4String , G4String fileName)
{

// Check there is a GatePhantom attached to this source

GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckGeometryAttached( GetCreator()->GetObjectName() );

if ( Ph != 0) 
{G4cout << " The Object "<< Ph->GetName()
<<" is attached to the "<<m_name<<" Geometry Voxel Reader"<<G4endl;
} 


  if (m_voxelTranslator == NULL) {
    G4cout << "GateGeometryVoxelImageReader::ReadFile: WARNING: Insert the translator before reading the image" << G4endl
    << "                                                 Reading aborted." << G4endl;
    return;
  }

  m_fileName = fileName;

  EmptyStore();

  std::ifstream inFile;
  G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4String materialName;
  G4int    imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  inFile >> nx >> ny >> nz;
  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;
  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  inFile >> dx >> dy >> dz;
  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;
  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
 inFile >> imageValue;
 materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
 if ( materialName != G4String("NULL") ) {
   G4Material* material = mMaterialDatabase.GetMaterial(materialName);
   AddVoxel(ix, iy, iz, material);
 } else {
   G4cout << "GateSourceVoxelImageReader::ReadFile: WARNING: voxel not added (material translation not found)" << G4endl;
 }
      }
    }
  }

  inFile.close();

  UpdateParameters();

  if (m_compressor) {
    Compress();
    G4cout << "GateGeometryVoxelImageReader::ReadFile: For your information, the voxel store has been emptied." << G4endl;
  }
 
}
