/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! \file 
  \brief Implementation of GateImageNestedParametrisation
*/
#include "GateImageNestedParametrisation.hh"
#include "GateImageNestedParametrisedVolume.hh"
#include "GateDetectorConstruction.hh"

#include "G4ThreeVector.hh"
//#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"

#include "GateMessageManager.hh"

//-----------------------------------------------------------------------------
GateImageNestedParametrisation::GateImageNestedParametrisation(GateImageNestedParametrisedVolume * volume)
  : pVolume(volume)
{

  GateMessage("Volume",5,"Begin GateImageNestedParametrisation()"<<G4endl);
  pVolume->BuildLabelToG4MaterialVector(mVectorLabel2Material);

  mAirMaterial = 
    GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air");

  // Computation of Z position
  G4double zp;
  G4double fNz = pVolume->GetImage()->GetResolution().z();
  GateMessage("Volume",6,"GateImageNestedParametrisation() -- fNz = " << fNz << G4endl);
  G4double fdZ = pVolume->GetImage()->GetVoxelSize().z()/2.0;
  GateMessage("Volume",6,"GateImageNestedParametrisation() -- fdZ = " << fdZ << G4endl);
  G4double fZoffset = 0; // ??
  for(int iz = 0; iz < fNz; iz++) {
    zp = (-fNz+1+2*iz)*fdZ+fZoffset;
    fpZ.push_back(zp);
    GateMessage("Volume",6,"GateImageNestedParametrisation() -- "<<iz << " -> " << zp << G4endl);
  }

  GateMessage("Volume",5,"End GateImageNestedParametrisation()"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateImageNestedParametrisation::~GateImageNestedParametrisation()
{
  GateMessageInc("Volume",5,"Begin ~GateImageNestedParametrisation()"<<G4endl);
  GateMessageDec("Volume",5,"End ~GateImageNestedParametrisation()"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageNestedParametrisation::ComputeTransformation(const G4int copyNo, G4VPhysicalVolume* physVol) const
{
  
  G4ThreeVector t(0., 0., fpZ[copyNo]);
  physVol->SetTranslation(t);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//void GateImageNestedParametrisation::ComputeDimensions(G4Box &,// voxels, 
//							 const G4int ,//c, 
//							 const G4VPhysicalVolume* //physVol) const
//						       ) const
//{
// Nothing to do here : always same size
//}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Material* GateImageNestedParametrisation::ComputeMaterial(G4VPhysicalVolume* /*physVol*/, // raise a warning if "release mode"
							    const G4int copyNo,
							    const G4VTouchable* parentTouch)
{
   if ( parentTouch == 0 ) { // FIXME replace NULL by 0. Done
    // GateDebugMessage("Volume",6,"GateImageNestedParametrisation::ComputeMaterial parent=0" 
    // 		     << physVol->GetName() << " copy=" << copyNo
    // 		     << " mat = " << copyNo
    // 		     << G4endl);
    return mVectorLabel2Material[0];
  }

  G4int ix = parentTouch->GetReplicaNumber(0);
  G4int iy = parentTouch->GetReplicaNumber(1);
  G4int iz = copyNo;
  
  GateDebugMessage("Volume",6,"GateImageNestedParametrisation::ComputeMaterial vox " 
		   << ix << " " << iy << " " << iz << G4endl);
  
  // Get label of material at voxel "copyNo"
  G4int lab = (G4int)pVolume->GetImage()->GetValue(ix, iy, iz);
  
  GateDebugMessage("Volume",6,"GateImageNestedParametrisation::ComputeMaterial lab " 
		   << lab << G4endl);
 
  // assert to be removed ? // Done.
  // assert(lab>=0);
  // assert(lab<(int)mVectorLabel2Material.size());
  
  // Get material from label
  G4Material* mat = mVectorLabel2Material[lab];
  
  GateDebugMessage("Volume",6,"mat = " << mat->GetName() << G4endl);
  
  return mat;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4int GateImageNestedParametrisation::GetNumberOfMaterials() const
{
  int nMat = mVectorLabel2Material.size();
  GateDebugMessage("Volume",6,"Nested GetNumberOfMaterials: "<<nMat << G4endl);
  return nMat;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4Material* GateImageNestedParametrisation::GetMaterial(G4int idx) const
{
  //  GateDebugMessage("Volume",6,"Nested GetMaterial " << idx << G4endl);
  return mVectorLabel2Material[idx];
}
//-----------------------------------------------------------------------------
