/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
#include "GateSourceVoxelInterfileReader.hh"
#include "GateSourceVoxelInterfileReaderMessenger.hh"
#include "GateVSourceVoxelTranslator.hh"
#include "GateMessageManager.hh"
#include <fstream>
#include <stdio.h>
#include <string.h>

GateSourceVoxelInterfileReader::GateSourceVoxelInterfileReader(GateVSource* source)
  : GateVSourceVoxelReader(source), GateInterfileHeader()
{
  GateError("GateSourceVoxelInterfileReader is obsolete, use GateSourceVoxelImageReader instead!");
  nVerboseLevel = 0;
  m_name = G4String("interfileReader");
  m_type = G4String("interfile");
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
      G4cout << "GateSourceVoxelInterfileReader::ReadFile: ERROR : insert a translator first\n";
      return;
  }
  G4cout << "------------------------------------------------------------------------------------------------\n"
         << "WARNING: Macro commands related to voxelized source description have been modified in GATE V7.1.\n"
         << "Older ones are being deprecated and will be removed from the next release.\n"
         << "Please, have a look at the related documentation at:\n"
         << "http://wiki.opengatecollaboration.org/index.php/Users_Guide_V7.1:Voxelized_Source_and_Phantom\n"
          << "------------------------------------------------------------------------------------------------\n";

  G4cout << "GateSourceVoxelInterfileReader::ReadFile : fileName: " <<  headerFileName << Gateendl;

  ReadHeader(headerFileName);

  std::vector<DefaultPixelType> buffer;

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
  SetArraySize(G4ThreeVector(nx, ny, nz));

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
	       << " is attached to the "<<m_name<<" Source Voxel Reader.\n";
  }  else {
      G4cout << " GateSourceVoxelInterfileReader::ReadFile   WARNING The Object "<< Ph->GetName()
	       << " is not attached to any Geometry Voxel Reader.\n";
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

  std::vector<DefaultPixelType> buffer;
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
  SetArraySize(G4ThreeVector(nx, ny, nz));

  for (G4int iz=0; iz<nz; iz++) {
      for (G4int iy=0; iy<ny; iy++) {
	  for (G4int ix=0; ix<nx; ix++) {
	      imageValue = buffer[ix+nx*iy+nx*ny*iz];
	      activity = m_voxelTranslator->TranslateToActivity(imageValue);
	      if (activity > 0.)
	    	  AddVoxel(ix, iy, iz, activity);
	  }
      }
  }
  PrepareIntegratedActivityMap();
}
