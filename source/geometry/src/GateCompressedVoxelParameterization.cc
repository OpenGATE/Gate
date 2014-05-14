/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateCompressedVoxelParameterization.hh"

#include "G4LogicalVolume.hh"
#include "G4VisAttributes.hh"

#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCutsTable.hh"

#include "GateDetectorConstruction.hh"
#include "GateMaterialDatabase.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "GateGeometryVoxelTabulatedTranslator.hh"
#include "GateGeometryVoxelRangeTranslator.hh"

//! Constructor.
GateCompressedVoxelParameterization::GateCompressedVoxelParameterization( 
						     GateVGeometryVoxelReader* voxR, 
						     const G4ThreeVector& voxN, 
						     const G4ThreeVector& voxS):
  GatePVParameterisation(),
  voxelNumber(voxN),
  voxelSize(voxS),
  voxelZeroPos( KroneckerProduct(voxelNumber-G4ThreeVector(1,1,1), voxelSize) / -2.0 ),
  voxelReader(voxR),
  voxelTranslator( voxelReader->GetVoxelTranslator() ),
  nxy( static_cast<int> ( voxelNumber.x() * voxelNumber.y() ) ),
  nx ( static_cast<int> ( voxelNumber.x() ) ){
  
  
}

G4int GateCompressedVoxelParameterization::GetNbOfCopies() {

GateVoxelCompressor* Compressor =  voxelReader->GetCompressorPtr();

if ( Compressor != 0 ) {return Compressor->GetNbOfCopies();} 
else {return G4int(voxelNumber.x() * voxelNumber.y() * voxelNumber.z());}

}
// ---------------------------------------------------------------------------------------
// Compute Transformation
// Sets the translation (offset) and rotation (always zero) of the voxel
// relative to center of the enclosing box

void GateCompressedVoxelParameterization::ComputeTransformation(G4int copyNo  ,G4VPhysicalVolume * pv) const{
  
  // Calculate the relative distance "relPos" of the center of the current voxel
  // from the center of the corner voxel (voxel #0), and
  // calculate the position (relative to the center of the matrix) by adding to relpos the corner position

  const GateCompressedVoxel& voxel( voxelReader->GetCompressor().GetVoxel(copyNo) );

  const G4ThreeVector location( voxel[2], voxel[1], voxel[0] );
  const G4ThreeVector size    ( voxel[5], voxel[4], voxel[3] );
  const G4ThreeVector unity   ( 1, 1, 1 );

  G4ThreeVector relPos( KroneckerProduct( location + (size-unity)/2.0, voxelSize) );
  G4ThreeVector xlat  ( relPos+voxelZeroPos );
 
  pv->SetTranslation( xlat );
  pv->SetRotation(0);

}

// ---------------------------------------------------------------------------------------
// Compute Dimensions
void GateCompressedVoxelParameterization::ComputeDimensions(G4Box& box, const G4int copyNo, const G4VPhysicalVolume* ) const{                                                   

  G4ThreeVector size;
  if ( voxelReader->GetCompressorPtr()->GetCompressionRatio() == 0 ) size = G4ThreeVector(voxelReader->GetVoxelNx(),voxelReader->GetVoxelNy(),voxelReader->GetVoxelNz());
  else
  {
   const GateCompressedVoxel& voxel( voxelReader->GetCompressor().GetVoxel(copyNo) );
  
   const G4ThreeVector size    ( voxel[5], voxel[4], voxel[3] );  
   G4ThreeVector halfDimensions( KroneckerProduct( size , voxelSize)/2.0 );

   box.SetXHalfLength( halfDimensions.x() );
   box.SetYHalfLength( halfDimensions.y() );
   box.SetZHalfLength( halfDimensions.z() );

  } 
}

// ---------------------------------------------------------------------------------------
//  Compute Material
//  Sets the visibility attributes of the voxel;
//  and return a pointer to its material
G4Material* GateCompressedVoxelParameterization::ComputeMaterial(G4int copyNo , G4VPhysicalVolume * pv , const G4VTouchable*)
{
  const GateCompressedVoxel& voxel( voxelReader->GetCompressor().GetVoxel(copyNo) );

  G4Material*      mp( (*G4Material::GetMaterialTable())[voxel[6]] );
  G4LogicalVolume* lv( pv->GetLogicalVolume()                  );
  
  if (voxelTranslator){
    lv->SetVisAttributes( voxelTranslator->GetMaterialAttributes(mp) );
  }

  return mp;

}
  
