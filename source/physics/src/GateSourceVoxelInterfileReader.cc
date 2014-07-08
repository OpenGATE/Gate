/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"
#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
#include "GateSourceVoxelInterfileReader.hh"
#include "GateSourceVoxelInterfileReaderMessenger.hh"
#include "GateVSourceVoxelTranslator.hh"
#include "GateMessageManager.hh"
#include <fstream>
#include <stdio.h>
#include <string.h>

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateSourceVoxelInterfileReader::GateSourceVoxelInterfileReader(GateVSource* source)
  : GateVSourceVoxelReader(source), GateInterfileHeader()
{
  nVerboseLevel = 0;
  m_name = G4String("interfileReader");
  m_messenger = new GateSourceVoxelInterfileReaderMessenger(this);
  m_fileName  = G4String("");
  IsFirstFrame = true;
}

GateSourceVoxelInterfileReader::~GateSourceVoxelInterfileReader()
{
  if (m_messenger) {
      delete m_messenger;
  }
}

void GateSourceVoxelInterfileReader::ReadFile(G4String headerFileName)
{
  if (!m_voxelTranslator) {
      G4cout << "GateSourceVoxelImageReader::ReadFile: ERROR : insert a translator first" << G4endl;
      return;
  }
  G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " <<  headerFileName << G4endl;

  ReadHeader(headerFileName);

  std::vector<PixelType> buffer;

  ReadData(buffer);

  G4double activity;
  G4double imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
      for (G4int iy=0; iy<ny; iy++) {
	  for (G4int ix=0; ix<nx; ix++) {
	      imageValue = buffer[ix+nx*iy+nx*ny*iz];
	      activity = m_voxelTranslator->TranslateToActivity(imageValue);
	      if (activity > 0.) {
		  AddVoxel(ix, iy, iz, activity);
	      }
	  }
      }
  }
  PrepareIntegratedActivityMap();
}

/* PY Descourt 08/09/2009 */

void GateSourceVoxelInterfileReader::ReadRTFile(G4String headerFileName, G4String dataFileName)
{
  Initialize();
  // Check if there is a GatePhantom attached
  GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckSourceAttached( m_name );

  if ( Ph != 0) {
      G4cout << " The Object "<< Ph->GetName()
	       << " is attached to the "<<m_name<<" Source Voxel Reader." << G4endl;
  }  else {
      G4cout << " GateSourceVoxelInterfileReader::ReadFile   WARNING The Object "<< Ph->GetName()
	       << " is not attached to any Geometry Voxel Reader."<<G4endl;
  }

  if (!m_voxelTranslator) {
      G4Exception("GateSourceVoxelInterfileReader::ReadFile", "NoTranslator", FatalException, "ERROR : insert a translator first");
  }

  if ( IsFirstFrame == true ) {
      ReadHeader(headerFileName);
      IsFirstFrame = false;
  }
  // override filename from header
  m_dataFileName = dataFileName;

  std::vector<PixelType> buffer;
  ReadData(m_dataFileName, buffer);

  G4double activity;
  G4double imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
      for (G4int iy=0; iy<ny; iy++) {
	  for (G4int ix=0; ix<nx; ix++) {
	      imageValue = buffer[ix+nx*iy+nx*ny*iz];
	      activity = m_voxelTranslator->TranslateToActivity(imageValue);
	      if (activity > 0.)
		AddVoxel_FAST(ix, iy, iz, activity);
	  }
      }
  }
  PrepareIntegratedActivityMap();
}
