/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVoxelBoxParameterization.hh"

#include "G4LogicalVolume.hh"
#include "G4VisAttributes.hh"

#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCutsTable.hh"

#include "GateMaterialDatabase.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "GateGeometryVoxelTabulatedTranslator.hh"
#include "GateGeometryVoxelRangeTranslator.hh"

//! Constructor.
GateVoxelBoxParameterization::GateVoxelBoxParameterization( 
						     GateVGeometryVoxelReader* voxR, 
						     const G4ThreeVector& voxN, 
						     const G4ThreeVector& voxS):
  GatePVParameterisation(),
  voxelNumber(voxN),
  voxelSize(voxS),
  voxelZeroPos( KroneckerProduct(voxelNumber-G4ThreeVector(1,1,1), voxelSize) / -2.0 ),
  voxelReader(voxR),
  translator( voxelReader->GetVoxelTranslator()),
  nxy( static_cast<int> ( voxelNumber.x() * voxelNumber.y() ) ),
  nx ( static_cast<int> ( voxelNumber.x() ) ){
    
}


// Compute Transformation
// Sets the translation (offset) and rotation (always zero) of the voxel
// relative to center of the enclosing box

void GateVoxelBoxParameterization::ComputeTransformation(G4int copyNo  ,G4VPhysicalVolume * pv) const
{
    
  // Calculate the relative distance "relPos" of the center of the current voxel
  // from the center of the corner voxel (voxel #0), and
  // calculate the position (relative to the center of the matrix) by adding to relpos the corner position
  G4ThreeVector relPos( KroneckerProduct( ComputeArrayIndices(copyNo), voxelSize) );
  G4ThreeVector xlat  ( relPos+voxelZeroPos );
 
  pv->SetTranslation( xlat );
  pv->SetRotation(0);
   
}


// Compute Dimensions (for documentation purposes only)
// The dimensions of the voxel box are set in GateVoxelBoxParameterized
// once and for all in ConstructGeometry.  Nothing needs to be done here

void GateVoxelBoxParameterization::ComputeDimensions(G4Box*, const G4int ,const G4VPhysicalVolume* ) const
{
}


//  Compute Material
//  Sets the visibility attributes of the voxel;
//  and return a pointer to its material



G4Material* GateVoxelBoxParameterization::ComputeMaterial(G4int copyNo , G4VPhysicalVolume * pv ){


  G4Material*      mp( voxelReader  ->GetVoxelMaterial(copyNo)  );
  G4LogicalVolume* lv( pv->GetLogicalVolume()                   );
  
  if (translator){
    lv->SetVisAttributes( translator->GetMaterialAttributes(mp) );
  }

  return mp;

}
