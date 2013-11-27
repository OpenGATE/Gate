/*----------------------
  GATE version name: gate_v6

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

#include "GateImageRegularParametrisedVolume.hh"
#include "GateImageRegularParametrisedVolumeMessenger.hh"
#include "GateDetectorConstruction.hh"
#include "GateImageNestedParametrisation.hh"
#include "GateMultiSensitiveDetector.hh"
#include "GateMiscFunctions.hh"

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
  GateMessageDec("Volume",5,"End GateImageRegularParametrisedVolume("<<name<<")"<<G4endl);
}
///---------------------------------------------------------------------------

///---------------------------------------------------------------------------
/// Destructor
GateImageRegularParametrisedVolume::~GateImageRegularParametrisedVolume()
{
  GateMessageInc("Volume",5,"Begin ~GateImageRegularParametrisedVolume()"<<G4endl);
  if (pMessenger) delete pMessenger;

  delete mLogVol;
  delete mPhysVol;
  delete mRegularParam;

  GateMessageDec("Volume",5,"End ~GateImageRegularParametrisedVolume()"<<G4endl);
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
  LoadImageMaterialsTable();

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


  /*
  G4PhantomParameterisation* param = new G4PhantomParameterisation();
  param->SetVoxelDimensions(GetImage()->GetVoxelSize().x/2.0,
                            GetImage()->GetVoxelSize().y/2.0,
                            GetImage()->GetVoxelSize().z/2.0);
  param->SetNoVoxel(GetImage()->GetResolution().x,
                    GetImage()->GetResolution().y,
                    GetImage()->GetResolution().z);
  pVolume->BuildLabelToG4MaterialVector(mVectorLabel2Material);
  param->SetMaterial(&mVectorLabel2Material[0]);
  param->SetMaterialIndices

    bBox
    box voxels
    G4PVParameterised(voxel, bb, param)


>
>   size_t* mateIDs = new size_t[xNo*yNo*zNo];
>    G4int Zahler =0;
>    for(G4int i=0;i<zNo;i++){
>       for(G4int j=0;j<yNo;j++){
>         for(G4int k=0;k<xNo;k++){
>           mateIDs[Zahler] = 0;
>           Zahler++;
>     }}}
>   wpParam->SetMaterialIndices( mateIDs );
  */


  // Create the main volume (bounding box)
  G4String boxname = GetObjectName() + "_solid";
  GateMessage("Volume", 4, "GateImageRegularParametrisedVolume -- Create Box halfSize  = "
              << GetHalfSize() << G4endl);
  pBoxSolid = new G4Box(GetSolidName(), GetHalfSize().x(), GetHalfSize().y(), GetHalfSize().z());
  pBoxLog = new G4LogicalVolume(pBoxSolid, mater, GetLogicalVolumeName());
  G4RotationMatrix *rotm = new G4RotationMatrix;
  G4ThreeVector pos(0.,0.,0.);
  pBoxPhys = new G4PVPlacement(rotm, pos, pBoxLog, boxname+"_phys", GetMotherLogicalVolume(), false, 1);
  GateMessage("Volume",4,"GateImageRegularParametrisedVolume -- Mother box created" << G4endl);

  // Create voxel volume
  pVoxelSolid = new G4Box(GetObjectName()+"_voxelsolid",
                          GetImage()->GetVoxelSize().x()/2.0,
                          GetImage()->GetVoxelSize().y()/2.0,
                          GetImage()->GetVoxelSize().z()/2.0);
  G4Material * Air =
    GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air");
  pVoxelLog = new G4LogicalVolume(pVoxelSolid, Air, GetObjectName()+"_voxelLog", 0,0,0);

  // Create the main Parametrisation
  GateMessage("Volume", 4, "GateImageRegularParametrisedVolume: create Parametrisation" << G4endl);
  /*mRegularParam = new GateImageRegularParametrisation(this);
  mRegularParam->BuildRegularParameterisation(mLogVol);
  mRegularParam->BuildContainerSolid(pBoxPhys);
  mRegularParam->SetSkipEqualMaterials(0);*/


  G4PhantomParameterisation* param = new G4PhantomParameterisation();
  param->SetVoxelDimensions(GetImage()->GetVoxelSize().x()/2.0,
                            GetImage()->GetVoxelSize().y()/2.0,
                            GetImage()->GetVoxelSize().z()/2.0);
  param->SetNoVoxel(GetImage()->GetResolution().x(),
                    GetImage()->GetResolution().y(),
                    GetImage()->GetResolution().z());
  BuildLabelToG4MaterialVector(mVectorLabel2Material);
  param->SetMaterials(mVectorLabel2Material);
  size_t* index = new size_t[GetImage()->GetNumberOfValues()];
  for(unsigned int i=0; i<GetImage()->GetNumberOfValues(); i++) {
    index[i] = GetImage()->GetValue(i);
  }
  param->SetMaterialIndices(index);
  param->SetSkipEqualMaterials(1);
  param->BuildContainerSolid(pBoxPhys);

  // Create the main Physical Volume G4PVParameterised
  GateMessage("Volume", 4, "GateImageRegularParametrisedVolume: create Physical Volume" << G4endl);
  mPhysVol = new G4PVParameterised(GetObjectName() + "_physVol",
                                   pVoxelLog, // logical volume for a voxel
                                   pBoxLog, // logical volume for the whole image
                                   kXAxis,
                                   GetImage()->GetNumberOfValues(), // number of pixels in the image
                                   param); // parametrisation
  mPhysVol->SetRegularStructureId(1);

  // Return the logical volume (will be pOwnLog);
  return pBoxLog;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::PrintInfo()
{
  // GateVImageVolume::PrintInfo();
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::PropagateGlobalSensitiveDetector()
{
  if (m_sensitiveDetector) {
    GatePhantomSD* phantomSD = GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD();
    pVoxelLog->SetSensitiveDetector(phantomSD);
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageRegularParametrisedVolume::PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector * msd)
{
  GateDebugMessage("Volume", 5, "Add SD to child" << G4endl);
  pVoxelLog->SetSensitiveDetector(msd);
}
//---------------------------------------------------------------------------
