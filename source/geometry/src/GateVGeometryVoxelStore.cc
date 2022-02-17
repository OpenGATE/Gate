/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVGeometryVoxelStore.hh"
#include "GateMaterialDatabase.hh"
#include "globals.hh"

#include "GateDetectorConstruction.hh"
#include "GateVVolume.hh"

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_GENERAL
#include "GateOutputMgr.hh"
#endif

GateVGeometryVoxelStore::GateVGeometryVoxelStore(GateVVolume* creator)
  : m_creator(creator)
    , m_voxelNx(0)
    , m_voxelNy(0)
    , m_voxelNz(0)
    , m_voxelSize(G4ThreeVector())
    , m_position(G4ThreeVector())
{
  m_position = G4ThreeVector();

  G4double voxelSize = 1.*mm;
  m_voxelSize = G4ThreeVector(voxelSize,voxelSize,voxelSize);

  SetDefaultMaterial( "G4_Galactic" );
}

GateVGeometryVoxelStore::~GateVGeometryVoxelStore()
{
}

void GateVGeometryVoxelStore::SetDefaultMaterial(G4String materialName) 
{ 
  SetDefaultMaterial( theMaterialDatabase.GetMaterial(materialName) );
}

void GateVGeometryVoxelStore::Describe(G4int ) 
{

  G4cout << "  Geom. voxel store -----> " << m_type << Gateendl
	 << "  position  (mm)         : " 
	 << GetPosition().x()/mm << " " 
	 << GetPosition().y()/mm << " " 
	 << GetPosition().z()/mm << Gateendl
	 << "  voxel size  (mm)       : " 
	 << GetVoxelSize().x()/mm << " " 
	 << GetVoxelSize().y()/mm << " " 
	 << GetVoxelSize().z()/mm << Gateendl;

}

void GateVGeometryVoxelStore::Dump() 
{
#ifdef G4ANALYSIS_USE_GENERAL
  // Here we fill the histograms through the OutputMgr manager
  GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
  outputMgr->RecordVoxels(this);
#endif
  
}
