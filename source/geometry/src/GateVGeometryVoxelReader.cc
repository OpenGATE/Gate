/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVGeometryVoxelReader.hh"
#include "GateDetectorConstruction.hh"

#include "GateVVolume.hh"

#include "GateVGeometryVoxelTranslator.hh"
#include "GateGeometryVoxelTabulatedTranslator.hh"
#include "GateGeometryVoxelRangeTranslator.hh"

GateVGeometryVoxelReader::GateVGeometryVoxelReader(GateVVolume* inserter)
  //  : GateGeometryVoxelMapStore(inserter)
  : GateGeometryVoxelArrayStore(inserter)
  , m_voxelTranslator(0)
  , m_fileName(G4String("NULL"))
{
  mMaterialDatabase = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase;
}

GateVGeometryVoxelReader::~GateVGeometryVoxelReader()
{
  if (m_voxelTranslator) {
    delete m_voxelTranslator;
  }
}

void GateVGeometryVoxelReader::Describe(G4int level) 
{
  G4cout << "  Geom. voxel reader ----> " << m_type << G4endl
	 << "  file name              : " << m_fileName << G4endl;

  //  GateGeometryVoxelMapStore::Describe(level);
  GateGeometryVoxelArrayStore::Describe(level);
}

void GateVGeometryVoxelReader::InsertTranslator(G4String translatorType)
{
  if (m_voxelTranslator) {
    G4cout << "GateVGeometryVoxelReader::InsertTranslator: voxel translator already defined" << G4endl;
  } else {
    if (translatorType == G4String("tabulated")) {
      m_voxelTranslator = new GateGeometryVoxelTabulatedTranslator(this);
    } else if (translatorType == G4String("range")) {
      m_voxelTranslator = new GateGeometryVoxelRangeTranslator(this);
    } else {
      G4cout << "GateVGeometryVoxelReader::InsertTranslator: unknown translator type" << G4endl;
    }
  }

}

void GateVGeometryVoxelReader::RemoveTranslator()
{
  if (m_voxelTranslator) {
    delete m_voxelTranslator;
    m_voxelTranslator = 0;
  } else {
    G4cout << "GateVGeometryVoxelReader::RemoveTranslator: voxel translator not defined" << G4endl;
  }
}
