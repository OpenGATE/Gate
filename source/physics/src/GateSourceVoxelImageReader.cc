/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"

#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
#include "GateSourceVoxelImageReader.hh"
#include "GateSourceVoxelImageReaderMessenger.hh"
#include "GateVSourceVoxelTranslator.hh"
//#include "fstream.h"
//LF
#include <fstream>
//LF

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateSourceVoxelImageReader::GateSourceVoxelImageReader(GateVSource* source)
  : GateVSourceVoxelReader(source)
{
  nVerboseLevel = 0;

  m_name = G4String("imageReader");

  m_messenger = new GateSourceVoxelImageReaderMessenger(this);
}

GateSourceVoxelImageReader::~GateSourceVoxelImageReader()
{
  delete m_messenger;
}

void GateSourceVoxelImageReader::ReadFile(G4String fileName)
{
  if (!m_voxelTranslator) {
    G4cout << "GateSourceVoxelImageReader::ReadFile: ERROR : insert a translator first" << G4endl;
    return;
  }

  std::ifstream inFile;
  G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4double activity;
  G4int imageValue;
  G4int nx, ny, nz;
  G4double dx, dy, dz;

  inFile >> nx >> ny >> nz;

  inFile >> dx >> dy >> dz;
  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
	inFile >> imageValue;
	activity = m_voxelTranslator->TranslateToActivity(imageValue);
	if (activity > 0.) {
	  AddVoxel(ix, iy, iz, activity);
	}
      }
    }
  }

  inFile.close();

  PrepareIntegratedActivityMap();

}


/* PY Descourt 11/12/2008 */ 
void GateSourceVoxelImageReader::ReadRTFile(G4String , G4String fileName)
{

// Check there is a GatePhantom attached to this source

GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckSourceAttached( m_name );

if ( Ph != 0) 
{G4cout << " The Object "<< Ph->GetName()
<<" is attached to the "<<m_name<<" Geometry Voxel Reader"<<G4endl;

} 


  if (!m_voxelTranslator) {
    G4cout << "GateSourceVoxelImageReader::ReadFile: ERROR : insert a translator first" << G4endl;
    return;
  }

  std::ifstream inFile;
  G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4double activity;
  G4int imageValue;
  G4int nx, ny, nz;
  G4double dx, dy, dz;

  inFile >> nx >> ny >> nz;

  inFile >> dx >> dy >> dz;
  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
 inFile >> imageValue;
 activity = m_voxelTranslator->TranslateToActivity(imageValue);
 if (activity > 0.) {
   AddVoxel(ix, iy, iz, activity);
 }
      }
    }
  }

  inFile.close();

  PrepareIntegratedActivityMap();

}
/* PY Descourt 11/12/2008 */ 
