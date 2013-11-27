/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! \file
  \brief Implementation of GateImageRegularParametrisation
*/

#include "GateImageRegularParametrisation.hh"
#include "GateImageRegularParametrisedVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateMessageManager.hh"

#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVParameterised.hh"
#include "G4PVPlacement.hh"

//-----------------------------------------------------------------------------
GateImageRegularParametrisation::GateImageRegularParametrisation(GateImageRegularParametrisedVolume * volume)
  :G4PhantomParameterisation(), pVolume(volume)
{
  GateMessage("Volume",5,"Begin GateImageRegularParametrisation()"<<G4endl);
  pVolume->BuildLabelToG4MaterialVector(mVectorLabel2Material);

  mAirMaterial =
    GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air");

  GateMessage("Volume",5,"End GateImageRegularParametrisation()"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImageRegularParametrisation::~GateImageRegularParametrisation()
{
  GateMessageInc("Volume",5,"Begin ~GateImageRegularParametrisation()"<<G4endl);
  GateMessageDec("Volume",5,"End ~GateImageRegularParametrisation()"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageRegularParametrisation::BuildRegularParameterisation(G4LogicalVolume * logVol)
{
  DD("GateImageRegularParametrisation::BuildRegularParameterisation");

  // Get the pointer to the image and set the sizes
  GateImage * im = pVolume->GetImage();
  SetVoxelDimensions( im->GetVoxelSize().x(), im->GetVoxelSize().y(), im->GetVoxelSize().z());
  SetNoVoxel( im->GetResolution().x(), im->GetResolution().y(), im->GetResolution().z());

  // Create physical volume
  // G4RotationMatrix *rotm = new G4RotationMatrix;
  // G4ThreeVector pos(0.,0.,0.);
  // G4PVPlacement * p = new G4PVPlacement(rotm, pos, // rotation, translation
  //                                       logVol, // log volume
  //                                       pVolume->GetLogicalVolumeName()+"_voxelphys", //name
  //                                       pVolume->GetMotherLogicalVolume(),
  //                                       false,// No op. bool.
  //                                       1);    // Copy number
  // BuildContainerSolid(p);

  //DS FIXME check if the voxels are completely filling the container volume

  DD("End BuildRegularParameterisation");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Material* GateImageRegularParametrisation::ComputeMaterial(const G4int copyNo,
                                                             G4VPhysicalVolume* pv,
                                                             const G4VTouchable* parentTouch)
{
  DD("ComputeMaterial");
  if ( parentTouch == 0 ) {
    // (called during init)
    //GateMessage("Volume",6,"GateImageRegularParametrisation::ComputeMaterial parentTouch=0"
    //           << pv->GetName() << " copy=" << copyNo << G4endl);
    return mVectorLabel2Material[0];
  }

  DD(copyNo);
  // Retrieve the voxel value
  G4int label = (G4int)pVolume->GetImage()->GetValue(copyNo);
  GateMessage("Volume", 6, "copy = " << copyNo << " label = " << label << G4endl);

  // Get material from label
  DD(mVectorLabel2Material.size());
  G4Material* mat = mVectorLabel2Material[label];

  GateMessage("Volume",6,"mat = " << mat->GetName() << G4endl);
  return mat;
}
//-----------------------------------------------------------------------------
