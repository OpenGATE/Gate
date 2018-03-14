/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "G4SystemOfUnits.hh"

#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
#include "GateSourceVoxelImageReader.hh"
#include "GateSourceVoxelImageReaderMessenger.hh"
#include "GateVSourceVoxelTranslator.hh"

//-----------------------------------------------------------------------------
GateSourceVoxelImageReader::GateSourceVoxelImageReader(GateVSource* source)
  : GateVSourceVoxelReader(source)
{
  nVerboseLevel = 0;
  m_name = G4String("imageReader");
  m_type = G4String("image");
  m_messenger = new GateSourceVoxelImageReaderMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateSourceVoxelImageReader::~GateSourceVoxelImageReader()
{
  delete m_messenger;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSourceVoxelImageReader::ReadFile(G4String filename)
{
  GateImage * image = new GateImage;

  if (!m_voxelTranslator) {
    GateError("GateSourceVoxelImageReader::ReadFile: ERROR : insert a translator first\n");
  }

  G4double activity;
  G4int nx, ny, nz;
  G4double vx, vy, vz;

  image->Read(filename);
  nx=image->GetResolution()[0];
  ny=image->GetResolution()[1];
  nz=image->GetResolution()[2];
  vx=image->GetVoxelSize()[0];
  vy=image->GetVoxelSize()[1];
  vz=image->GetVoxelSize()[2];

  SetVoxelSize( G4ThreeVector(vx, vy, vz) * mm );
  SetArraySize(G4ThreeVector(nx, ny, nz));

  m_image_origin = image->GetOrigin();

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
        PixelType imageValue = image->GetValue(ix, iy, iz);
        activity = m_voxelTranslator->TranslateToActivity(imageValue);
        if (activity > 0) {
          AddVoxel(ix, iy, iz, activity);
        }
      }
    }
  }
  PrepareIntegratedActivityMap();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSourceVoxelImageReader::ReadRTFile(G4String , G4String fileName)
{

  // Check there is a GatePhantom attached to this source

  GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckSourceAttached( m_name );

  if ( Ph != 0)
    {G4cout << " The Object "<< Ph->GetName()
            <<" is attached to the "<<m_name<<" Geometry Voxel Reader\n";

    }


  if (!m_voxelTranslator) {
    G4cout << "GateSourceVoxelImageReader::ReadFile: ERROR : insert a translator first\n";
    return;
  }

  std::ifstream inFile;
  G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " << fileName << Gateendl;
  inFile.open(fileName.c_str(),std::ios::in);

  G4double activity;
  G4int imageValue;
  G4int nx, ny, nz;
  G4double dx, dy, dz;

  inFile >> nx >> ny >> nz;

  inFile >> dx >> dy >> dz;
  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );
  SetArraySize(G4ThreeVector(nx, ny, nz));

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
//-----------------------------------------------------------------------------
