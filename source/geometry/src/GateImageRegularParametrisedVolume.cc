/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*! /file
  /brief Implementation of GateImageRegularParametrisedVolume
*/

#include "G4PVParameterised.hh"
#include "G4PVPlacement.hh"
#include "G4Box.hh"
#include "G4VoxelLimits.hh"
#include "G4TransportationManager.hh"
#include "G4RegionStore.hh"

#include "GateImageRegularParametrisedVolumeMessenger.hh"
#include "GateImageRegularParametrisedVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateImageNestedParametrisation.hh"
#include "GateMultiSensitiveDetector.hh"
#include "GateMiscFunctions.hh"
#include "GateImageBox.hh"

///---------------------------------------------------------------------------
/// Constructor with :
/// the path to the volume to create (for commands)
/// the name of the volume to create
/// Creates the messenger associated to the volume
GateImageRegularParametrisedVolume::GateImageRegularParametrisedVolume(const G4String& name,
								     G4bool acceptsChildren,
								     G4int depth)
  : GateVImageVolume(name,acceptsChildren,depth)
{
  GateMessageInc("Volume",5,"Begin GateImageRegularParametrisedVolume("<<name<<")"<<G4endl);
  pMessenger = new GateImageRegularParametrisedVolumeMessenger(this);
  SetSkipEqualMaterialsFlag(false);
  GateMessageDec("Volume",5,"End GateImageRegularParametrisedVolume("<<name<<")"<<G4endl);
}
///---------------------------------------------------------------------------


///---------------------------------------------------------------------------
/// Destructor
GateImageRegularParametrisedVolume::~GateImageRegularParametrisedVolume()
{
  GateMessageInc("Volume",5,"Begin ~GateImageRegularParametrisedVolume()"<<G4endl);
  if (pMessenger) delete pMessenger;

  delete mImagePhysVol;
  delete mVoxelSolid;
  delete mVoxelLog;
  delete mImageData;

  GateMessageDec("Volume",5,"End ~GateImageRegularParametrisedVolume()"<<G4endl);
}
///---------------------------------------------------------------------------

///---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::SetSkipEqualMaterialsFlag(bool b)
{
G4cout<<"### WARNING ### setSkipEqualMaterials at false !! The Geant4 method is not safe since the release 9.5 - Need to be fixed"<<G4endl;
  mSkipEqualMaterialsFlag = b;
}
///---------------------------------------------------------------------------


///---------------------------------------------------------------------------
bool GateImageRegularParametrisedVolume::GetSkipEqualMaterialsFlag() {
  return mSkipEqualMaterialsFlag;
}
///---------------------------------------------------------------------------


///---------------------------------------------------------------------------
/// Constructs
G4LogicalVolume* GateImageRegularParametrisedVolume::ConstructOwnSolidAndLogicalVolume(G4Material* mater,
										      G4bool /*flagUpdateOnly*/)
{
  GateMessageInc("Volume",3,"Begin GateImageRegularParametrisedVolume::ConstructOwnSolidAndLogicalVolume()" << G4endl);
  // Load image and material table (false = no additional border)
  LoadImage(false);

  // Cheat : if needed, for visu purpose, only create 1x2x3 voxels
  if (mIsBoundingBoxOnlyModeEnabled) {
    // Create few pixels
    G4ThreeVector r(1,2,3);
    G4ThreeVector s(GetImage()->GetSize().x()/1.0,
                    GetImage()->GetSize().y()/2.0,
                    GetImage()->GetSize().z()/3.0);
    GetImage()->SetResolutionAndVoxelSize(r, s);
  }

  // Set position if IsoCenter is Set
  UpdatePositionWithIsoCenter();

  // Create the main volume (bounding box)
  G4String boxname = GetObjectName() + "_solid";
  pBoxSolid = new GateImageBox(*GetImage(), GetSolidName());
  pBoxLog = new G4LogicalVolume(pBoxSolid, mater, GetLogicalVolumeName());

  LoadImageMaterialsTable();

  //FIXME position
  G4RotationMatrix *rotm = new G4RotationMatrix;
  G4ThreeVector pos(0.,0.,0.);
  pBoxPhys = new G4PVPlacement(rotm, pos, pBoxLog, boxname+"_phys", GetMotherLogicalVolume(), false, 1);

  // Create voxel volume (default material = Vacuum
  mVoxelSolid = new G4Box(GetObjectName()+"_voxelsolid",
                          GetImage()->GetVoxelSize().x()/2.0,
                          GetImage()->GetVoxelSize().y()/2.0,
                          GetImage()->GetVoxelSize().z()/2.0);
  G4Material * Vacuum =
    GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Vacuum");
  mVoxelLog = new G4LogicalVolume(mVoxelSolid, Vacuum, GetObjectName()+"_voxelLog", 0,0,0);

  // Create the main Parametrisation
  G4PhantomParameterisation* param = new G4PhantomParameterisation();
  param->SetVoxelDimensions(GetImage()->GetVoxelSize().x()/2.0,
                            GetImage()->GetVoxelSize().y()/2.0,
                            GetImage()->GetVoxelSize().z()/2.0);
  param->SetNoVoxel(GetImage()->GetResolution().x(),
                    GetImage()->GetResolution().y(),
                    GetImage()->GetResolution().z());
  BuildLabelToG4MaterialVector(mVectorLabel2Material);
  param->SetMaterials(mVectorLabel2Material);
  // Convert image voxel into size_t type.
  mImageData = new size_t[GetImage()->GetNumberOfValues()];
  for(int i=0; i<GetImage()->GetNumberOfValues(); i++) {
    mImageData[i] = GetImage()->GetValue(i);
  }
  param->SetMaterialIndices(mImageData);
  param->SetSkipEqualMaterials(mSkipEqualMaterialsFlag);
  param->BuildContainerSolid(pBoxPhys);

  // Create the main Physical Volume G4PVParameterised
  GateMessage("Volume", 4, "GateImageRegularParametrisedVolume: create Physical Volume" << G4endl);
  mImagePhysVol = new G4PVParameterised(GetObjectName() + "_physVol",
                                   mVoxelLog, // logical volume for a voxel
                                   pBoxLog, // logical volume for the whole image
                                   kXAxis,
                                   GetImage()->GetNumberOfValues(), // number of pixels in the image
                                   param); // parametrisation
  mImagePhysVol->SetRegularStructureId(1);

  // Return the logical volume (will be pOwnLog);
  return pBoxLog;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::PrintInfo()
{
  GateMessage("Actor", 1, "GateImageRegularParametrisedVolume Actor " << G4endl);
  GateVImageVolume::PrintInfo();
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::PropagateGlobalSensitiveDetector()
{
  if (m_sensitiveDetector) {
    GatePhantomSD* phantomSD = GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD();
    mVoxelLog->SetSensitiveDetector(phantomSD);
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector * msd)
{
  GateDebugMessage("Volume", 5, "Add SD to child" << G4endl);
  mVoxelLog->SetSensitiveDetector(msd);
}
//---------------------------------------------------------------------------
