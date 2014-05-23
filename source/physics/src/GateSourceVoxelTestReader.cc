/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"

#include "GateSourceVoxelTestReader.hh"
#include "GateSourceVoxelTestReaderMessenger.hh"
//LF
//#include "fstream.h"
#include <fstream>
//LF

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateSourceVoxelTestReader::GateSourceVoxelTestReader(GateVSource* source)
  : GateVSourceVoxelReader(source)
{
  m_name = G4String("testReader");

  m_messenger = new GateSourceVoxelTestReaderMessenger(this);
}

GateSourceVoxelTestReader::~GateSourceVoxelTestReader()
{
  delete m_messenger;
}

void GateSourceVoxelTestReader::ReadFile(G4String fileName)
{
  // here begins the part for the specific file format:
  // open the file, read the parameters, decides if it is needed to create a new voxel or not, how, etc.

  // open the file
  std::ifstream inFile;
  G4cout << "GateSourceVoxelTestReader::ReadFile : fileName: " << fileName << G4endl;
  inFile.open(fileName.c_str(),std::ios::in);

  // read the parameters from the first lines of the file
  G4int nVoxels;
  G4int ix, iy, iz;
  G4double activity;
  inFile >> nVoxels;
  G4cout << "GateSourceVoxelTestReader::ReadFile : nVoxels: " << nVoxels << G4endl;

  // read the list of voxels
  for (G4int iV = 0; iV< nVoxels; iV++) {
    inFile >> ix >> iy >> iz >> activity;
    G4cout << "GateSourceVoxelTestReader::ReadFile : index: " 
	   << ix << " " << iy << " " << iz 
	   << " activity " << activity << G4endl;

    // create a new voxel only if the corresponding activity is > 0.
    if (activity > 0.) {
      AddVoxel(ix, iy, iz, activity*becquerel);
    }

  }

  inFile.close();

  PrepareIntegratedActivityMap();

}


void GateSourceVoxelTestReader::ReadRTFile(G4String /*header_fileName //WARNING: parameter not used */, G4String /*dataFileName //WARNING: parameter not used */)
{;}
