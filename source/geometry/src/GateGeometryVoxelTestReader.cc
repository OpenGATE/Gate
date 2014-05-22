/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <fstream>

#include "GateGeometryVoxelTestReader.hh"
#include "GateGeometryVoxelTestReaderMessenger.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "GateMaterialDatabase.hh"
#include "GateVVolume.hh"
//E#include "GateVoxelReplicaMatrix.hh"

GateGeometryVoxelTestReader::GateGeometryVoxelTestReader(GateVVolume* inserter)
  : GateVGeometryVoxelReader(inserter)
{
  m_name = G4String("testReader");
  m_fileName  = G4String("");
  m_messenger = new GateGeometryVoxelTestReaderMessenger(this);
}

GateGeometryVoxelTestReader::~GateGeometryVoxelTestReader()
{
  if (m_messenger) {
    delete m_messenger;
  }
}


void GateGeometryVoxelTestReader::Describe(G4int level) 
{
  G4cout << " Voxel reader type ---> " << m_name << G4endl;

  GateVGeometryVoxelReader::Describe(level);
  
}

void GateGeometryVoxelTestReader::ReadFile(G4String fileName)
{
  if (m_voxelTranslator == NULL) {
    G4cout << "GateGeometryVoxelTestReader::ReadFile: WARNING: Insert the translator before reading the image" << G4endl
	   << "                                                Reading aborted." << G4endl;
    return;
  }

  m_fileName = fileName;

  EmptyStore();

  std::ifstream inFile;
  G4cout << "GateGeometryVoxelTestReader::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4String materialName;
  G4int    imageValue;

  G4int    ix, iy, iz;

  G4int    nx, ny, nz;
  G4double dx, dy, dz;
  G4int    nTotVox;

  inFile >> nx >> ny >> nz;
  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;
  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  inFile >> dx >> dy >> dz;
  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;
  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  inFile >> nTotVox;
  G4cout << "nTotVox: " << nTotVox << G4endl;

  for (G4int iVox=0; iVox<nTotVox; iVox++) {

    inFile >> ix >> iy >> iz;
    inFile >> imageValue;

    G4cout << "  ix iy iz imageValue: " << ix << " " << iy << " " << iz << " " << imageValue << G4endl;

    materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
    if ( materialName != G4String("NULL") ) {
      G4Material* material = mMaterialDatabase.GetMaterial(materialName);
      AddVoxel(ix, iy, iz, material);
    } else {
      G4cout << "GateSourceVoxelTestReader::ReadFile: WARNING: voxel not added (material translation not found)" << G4endl;
    }

  }

  inFile.close();

  UpdateParameters();

//E  ((GateVoxelReplicaMatrix*)m_creator)->ResizeAll();
}

