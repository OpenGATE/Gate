/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*! /file 
  /brief Implementation of GateImageNestedParametrisedVolume
*/

#include "G4PVParameterised.hh"
#include "G4PVPlacement.hh"
#include "G4Box.hh"
#include "G4VoxelLimits.hh"
#include "G4TransportationManager.hh"
#include "G4RegionStore.hh"

#include "GateImageNestedParametrisedVolume.hh"
#include "GateImageNestedParametrisedVolumeMessenger.hh"
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
GateImageNestedParametrisedVolume::GateImageNestedParametrisedVolume(const G4String& name,
								     G4bool acceptsChildren,
								     G4int depth)
  : GateVImageVolume(name,acceptsChildren,depth)
{
  GateMessageInc("Volume",5,"Begin GateImageNestedParametrisedVolume("<<name<<")"<<G4endl);
  pMessenger = new GateImageNestedParametrisedVolumeMessenger(this);
  GateMessageDec("Volume",5,"End GateImageNestedParametrisedVolume("<<name<<")"<<G4endl);
}
///---------------------------------------------------------------------------

///---------------------------------------------------------------------------
/// Destructor 
GateImageNestedParametrisedVolume::~GateImageNestedParametrisedVolume()
{
  GateMessageInc("Volume",5,"Begin ~GateImageNestedParametrisedVolume()"<<G4endl);
  if (pMessenger) delete pMessenger;
  delete mVoxelParametrisation;
  delete mPhysVolX;
  delete mPhysVolY;
  delete mPhysVolZ;
  delete logXRep;
  delete logYRep;
  delete logZRep;
  GateMessageDec("Volume",5,"End ~GateImageNestedParametrisedVolume()"<<G4endl);
}
///---------------------------------------------------------------------------


///---------------------------------------------------------------------------
/// Constructs
G4LogicalVolume* GateImageNestedParametrisedVolume::ConstructOwnSolidAndLogicalVolume(G4Material* mater, 
										      G4bool /*flagUpdateOnly*/)
{
  GateMessageInc("Volume",3,"Begin GateImageNestedParametrisedVolume::ConstructOwnSolidAndLogicalVolume()" << G4endl);
  //---------------------------
  LoadImage(false);

  //---------------------------
  if (mIsBoundingBoxOnlyModeEnabled) {
    // Create few pixels
    G4ThreeVector r(1,2,3);
    G4ThreeVector s(GetImage()->GetSize().x()/1.0, 
                    GetImage()->GetSize().y()/2.0, 
                    GetImage()->GetSize().z()/3.0);
    GetImage()->SetResolutionAndVoxelSize(r, s);
    //    GetImage()->Fill(0);
    // GetImage()->PrintInfo();
  }

  //---------------------------
  // Set position if IsoCenter is Set
  UpdatePositionWithIsoCenter();

  //---------------------------
  // Mother volume
  G4String boxname = GetObjectName() + "_solid";
  GateMessage("Volume",4,"GateImageNestedParametrisedVolume -- Create Box halfSize  = " << GetHalfSize() << G4endl);

  pBoxSolid = new GateImageBox(*GetImage(), GetSolidName());
  pBoxLog = new G4LogicalVolume(pBoxSolid, mater, GetLogicalVolumeName()); 
  GateMessage("Volume",4,"GateImageNestedParametrisedVolume -- Mother box created" << G4endl);
  //---------------------------

  LoadImageMaterialsTable();
  
  //---------------------------
  GateMessageDec("Volume",4,"GateImageNestedParametrisedVolume -- Voxels construction" << G4endl);  
  G4ThreeVector voxelHalfSize = GetImage()->GetVoxelSize()/2.0;
  G4ThreeVector voxelSize = GetImage()->GetVoxelSize();
  GateMessage("Volume",4,"Create Voxel halfSize  = " << voxelHalfSize << G4endl);

  GateMessage("Volume",4,"Create vox*res x = " << voxelHalfSize.x()*GetImage()->GetResolution().x() << G4endl);
  GateMessage("Volume",4,"Create vox*res y = " << voxelHalfSize.y()*GetImage()->GetResolution().y() << G4endl);
  GateMessage("Volume",4,"Create vox*res z = " << voxelHalfSize.z()*GetImage()->GetResolution().z() << G4endl);

  //---------------------------
  // Parametrisation Y 
  G4String voxelYSolidName(GetObjectName() + "_voxel_solid_Y");
  G4VSolid* voxelYSolid =
    new G4Box(voxelYSolidName,
	      GetHalfSize().x(),
	      voxelHalfSize.y(),
	      GetHalfSize().z());
  G4String voxelYLogName(GetObjectName() + "_voxel_log_Y");
  // G4LogicalVolume*
  logYRep =
    new G4LogicalVolume(voxelYSolid,
			GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air"),
			voxelYLogName);
  G4String voxelYPVname = GetObjectName() + "_voxel_phys_Y";

  // G4VPhysicalVolume * pvy =
  mPhysVolY =
    new G4PVReplica(voxelYPVname,
		    logYRep,
		    pBoxLog, 
		    kYAxis,
		    (int)lrint(GetImage()->GetResolution().y()),
		    voxelSize.y());  
  GateMessage("Volume",4,"Create Nested Y : " << G4endl);
  GateMessage("Volume",4,"           nb Y : " << GetImage()->GetResolution().y() << G4endl);
  GateMessage("Volume",4,"          sizeY : " << voxelHalfSize.y() << G4endl);
  //---------------------------
  // Parametrisation X
  G4String voxelXSolidName(GetObjectName() + "_voxel_solid_X");
  G4VSolid* voxelXSolid =
    new G4Box(voxelXSolidName,
	      voxelHalfSize.x(),
	      voxelHalfSize.y(),
	      GetHalfSize().z());
  G4String voxelXLogName(GetObjectName() + "_voxel_log_X");
  // G4LogicalVolume*
  logXRep =
    new G4LogicalVolume(voxelXSolid,
			GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air"),
			voxelXLogName);
  G4String voxelXPVname = GetObjectName() + "_voxel_phys_X";



  // G4VPhysicalVolume * pvx =
  mPhysVolX =
    new G4PVReplica(voxelXPVname,
		    logXRep,
		    logYRep,
		    kXAxis,
		    (int)lrint(GetImage()->GetResolution().x()),
		    voxelSize.x()); 

  GateMessage("Volume",4,"Create Nested X : " << G4endl);
  GateMessage("Volume",4,"           nb X : " << GetImage()->GetResolution().x() << G4endl);
  GateMessage("Volume",4,"          sizeX : " << voxelHalfSize.x() << G4endl);

  //---------------------------
  // Parametrisation Z and NestedParameterisation
  G4String voxelZSolidName(GetObjectName() + "_voxel_solid_Z");
  G4VSolid* voxelZSolid =
    new G4Box(voxelZSolidName,
	      voxelHalfSize.x(),
	      voxelHalfSize.y(),
	      voxelHalfSize.z());
  G4String voxelZLogName(GetObjectName() + "_voxel_log_Z");
  // G4LogicalVolume*
  logZRep = 
    new G4LogicalVolume(voxelZSolid,
                        GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air"),
			voxelZLogName);
  
  G4VPVParameterisation * voxelParam = 
    mVoxelParametrisation = new GateImageNestedParametrisation(this);
  G4String voxelZPVname = GetObjectName() + "_voxel_phys_Z";

  // G4VPhysicalVolume * pvz = 
  mPhysVolZ =
    new G4PVParameterised(voxelZPVname,  // name
			  logZRep,      // logical volume
			  logXRep,      // Mother logical volume
			  kUndefined, // use kUndefined for 3D optimisation
			  (int)lrint(GetImage()->GetResolution().z()), // Number of copies = number of voxels
			  voxelParam);

  GateMessageInc("Volume",3,"End GateImageNestedParametrisedVolume::ConstructOwnSolidAndLogicalVolume()" << G4endl);
  return pBoxLog;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateImageNestedParametrisedVolume::PrintInfo()
{
  // GateVImageVolume::PrintInfo();
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
/// Compute and return a PhysVol with properties (dimension,
/// transformation and material) of the voxel 'index'. Used by the
/// optimized navigator
void GateImageNestedParametrisedVolume::GetPhysVolForAVoxel(const G4int index,
							    const G4VTouchable&, 
							    G4VPhysicalVolume ** ppPhysVol, 
							    G4NavigationHistory & history) const
{
  G4VPhysicalVolume * pPhysVol = (*ppPhysVol);
  
  GateDebugMessage("Volume",5,"GetPhysVolForAVoxel -- index = " << index << G4endl);
  if (pPhysVol) {
    GateDebugMessage("Volume",5,"GetPhysVolForAVoxel -- phys  = " << pPhysVol->GetName() << G4endl);
  }
  else {
    GateError("pPhysVol is null ?? ");
  }

  G4VSolid * pSolid = 0;	
  //G4Material * pMat = 0;
  // Get solid (no computation)
  pSolid = mVoxelParametrisation->ComputeSolid(index, pPhysVol);

  if (!pSolid) {
    GateError("pSolid is null ?? ");
  }
  
  // Get dimension (no computation, because same size)
  pSolid->ComputeDimensions(mVoxelParametrisation, index, pPhysVol);
  // Compute position// --> slow ?! should be precomputed ?
  G4ThreeVector v = GetImage()->GetCoordinatesFromIndex(index);
  int ix,iy,iz;
  ix = (int)lrint(v[0]);
  iy = (int)lrint(v[1]);
  iz = (int)lrint(v[2]);
  GateDebugMessage("Volume",5,"GetPhysVolForAVoxel -- voxel = " << v << G4endl);
  mVoxelParametrisation->ComputeTransformation(iz, pPhysVol);
  GateDebugMessage("Volume",5,"GetPhysVolForAVoxel -- phys T = " << pPhysVol->GetTranslation() << G4endl);

  // Set index
  mPhysVolX->SetCopyNo(ix);
  mPhysVolY->SetCopyNo(iy);
  mPhysVolZ->SetCopyNo(iz);

  GateDebugMessage("Volume",6," fHistory.GetTopVolume()->GetName() "
		   << history.GetTopVolume()->GetName() << G4endl);
  
  GateDebugMessage("Volume",6," fHistory.GetTopReplicaNo() "
		   << history.GetTopReplicaNo() << G4endl);
  
  if (history.GetTopVolume() != mPhysVolX) {
    history.NewLevel(mPhysVolX, kReplica, ix);
    G4TouchableHistory t(history);
    GateDebugMessage("Volume",6," fHistory.GetTopVolume()->GetName() "
		     << history.GetTopVolume()->GetName() << G4endl);
    
    GateDebugMessage("Volume",6," fHistory.GetTopReplicaNo() "
		     << history.GetTopReplicaNo() << G4endl);
    //pMat = mVoxelParametrisation->ComputeMaterial(iz, pPhysVol, &t);
  }
  else {
    // Get material
    //pMat = mVoxelParametrisation->ComputeMaterial(iz, pPhysVol, &pTouchable);
  }
  // Update history
  history.NewLevel(pPhysVol, kParameterised, iz);
  
  GateDebugMessage("Volume",5,"Trans = " << mPhysVolX->GetTranslation() << G4endl);
  GateDebugMessage("Volume",5,"Trans = " << mPhysVolY->GetTranslation() << G4endl);
  GateDebugMessage("Volume",5,"Trans = " << mPhysVolZ->GetTranslation() << G4endl);
  //GateDebugMessage("Volume",5,"Mat   = " << pMat->GetName() << G4endl);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageNestedParametrisedVolume::PropagateGlobalSensitiveDetector()
{
  if (m_sensitiveDetector) {
    GatePhantomSD* phantomSD = GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD();
    logYRep->SetSensitiveDetector(phantomSD);
    logXRep->SetSensitiveDetector(phantomSD);
    logZRep->SetSensitiveDetector(phantomSD);
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateImageNestedParametrisedVolume::PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector * msd)
{
  GateDebugMessage("Volume", 5, "Add SD to child" << G4endl);
  logXRep->SetSensitiveDetector(msd);
  logYRep->SetSensitiveDetector(msd);
  logZRep->SetSensitiveDetector(msd);
}
//---------------------------------------------------------------------------
