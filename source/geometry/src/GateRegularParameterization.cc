/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateRegularParameterization.hh"
#include "GateRegularParameterized.hh"
#include "GateFictitiousVoxelMapParameterized.hh"

#include "GateDetectorConstruction.hh"

#include "G4Material.hh"
#include "GateMaterialDatabase.hh"
#include "G4VisAttributes.hh"

#include "GateGeometryVoxelTabulatedTranslator.hh"
#include "GateGeometryVoxelRangeTranslator.hh"

#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"

///////////////////
//  Constructor  //
///////////////////

GateRegularParameterization::GateRegularParameterization( GateRegularParameterized* itsInserter , const G4ThreeVector& voxN )
 : G4PhantomParameterisation(),
   globalInserter(itsInserter),
   voxelReader(globalInserter->GetReader()),
   translator(voxelReader->GetVoxelTranslator()),
   voxelSize(voxelReader->GetVoxelSize()),
   voxelNumber(voxN)
{
  globalFictInserter=NULL;
  if (globalInserter->GetVerbosity()>=1) {
    G4cout << "+-+- Entering GateRegularParameterization::Constructor ..."
           << G4endl << std::flush;
  }
}

GateRegularParameterization::GateRegularParameterization ( GateFictitiousVoxelMapParameterized* itsInserter , const G4ThreeVector& voxN )
                : G4PhantomParameterisation(),
                globalFictInserter ( itsInserter ),
                voxelReader ( globalFictInserter->GetReader() ),
                translator ( voxelReader->GetVoxelTranslator() ),
                voxelSize ( voxelReader->GetVoxelSize() ),
                voxelNumber ( voxN )
{
  globalInserter=NULL;
  if ( globalFictInserter->GetVerbosity() >=1 ) {
    G4cout << "+-+- Entering GateRegularParameterization::Constructor ..."
           << G4endl << std::flush;
  }
}


////////////////////////////////////
//  BuildRegularParameterization  //
////////////////////////////////////

void GateRegularParameterization::BuildRegularParameterization()
{
  if (globalInserter!=NULL) {
    if (globalInserter->GetVerbosity()>=1) {
      G4cout << "++++ Entering GateRegularParameterization::BuildRegularParameterization ..."
             << G4endl << std::flush;
    }
  }
  else {
    if (globalFictInserter->GetVerbosity()>=1) {
      G4cout << "++++ Entering GateRegularParameterization::BuildRegularParameterization ..."
             << G4endl << std::flush;
    }
  }
  //---------------------------------------------------------------------------------------------//

  // Set the voxels dimensions
  G4double halfX = voxelSize.x()/2.;
  G4double halfY = voxelSize.y()/2.;
  G4double halfZ = voxelSize.z()/2.;
  SetVoxelDimensions( halfX, halfY, halfZ );

  // Set the number of voxels in each dimension
  G4int nVoxelX = voxelReader->GetVoxelNx();
  G4int nVoxelY = voxelReader->GetVoxelNy();
  G4int nVoxelZ = voxelReader->GetVoxelNz();
  SetNoVoxel( nVoxelX, nVoxelY, nVoxelZ );

  //---------------------------------------------------------------------------------------------//

  // We create a vector that will contain the list of all materials present in the parameterized box
  std::vector<G4String> mat;
  // Then we call the GetCompleteListOfMaterial method (of GateVGeometryVoxelTranslator)
  // that fills these vector
  translator->GetCompleteListOfMaterials(mat);

  // Now we build the different G4Material that we put in a vector
  std::vector<G4Material*> theMaterials;
  for (size_t nbMat=0; nbMat<mat.size(); nbMat++)
    theMaterials.push_back( GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(mat[nbMat]) );
  SetMaterials( theMaterials );

  // And then we search for each voxel his Material index (in the vector bellow)
  size_t nbVoxels = nVoxelX*nVoxelY*nVoxelZ;
  size_t* mateIDs = new size_t[nbVoxels];
  for (size_t copyNo=0; copyNo<nbVoxels; copyNo++) {
    G4String voxelMaterialName = voxelReader->GetVoxelMaterial(copyNo)->GetName();
    for (size_t indice=0; indice<mat.size(); indice++) {
      if (mat[indice]==voxelMaterialName)
      {
        mateIDs[copyNo]=indice;
        break;
      }
    }
  }
  SetMaterialIndices( mateIDs );

  //---------------------------------------------------------------------------------------------//

  // We create the physical volume that will contain all the voxels
  G4RotationMatrix *rotm = new G4RotationMatrix;
  G4ThreeVector pos(0.,0.,0.);

  if (globalInserter!=NULL) {
    cont_phys =
      new G4PVPlacement(rotm,                                             // Rotation : default
                        pos,                                              // Translation : default
                        globalInserter->GetCreator()->GetLogicalVolume(), // The logical volume
                        globalInserter->GetName()+"_phys",                // Name
                        globalInserter->GetMotherLogicalVolume(),         // Mother
                        false,                                            // No op. bool.
                        1);                                               // Copy number
  }
  else {
    cont_phys =
      new G4PVPlacement(rotm,                                                 // Rotation : default
                        pos,                                                  // Translation : default
                        globalFictInserter->GetCreator()->GetLogicalVolume(), // The logical volume
                        globalFictInserter->GetName()+"_phys",                // Name
                        globalFictInserter->GetMotherLogicalVolume(),         // Mother
                        false,                                                // No op. bool.
                        1);                                                   // Copy number
  }

  //---------------------------------------------------------------------------------------------//

  // We assign it as the container volume of the regular parameterization
  BuildContainerSolid(cont_phys);
  G4int doWeSkipEqualMaterials=1;
  if (globalInserter!=NULL) {
    doWeSkipEqualMaterials = globalInserter->GetSkipEqualMaterials();
  }
  else {
    doWeSkipEqualMaterials = globalFictInserter->GetSkipEqualMaterials();
  }
  SetSkipEqualMaterials(doWeSkipEqualMaterials);

  // And we check if the voxels are completely filling the container volume
  // Modifs Seb 20/03/2009
  G4double halfDimX = halfX*nVoxelX;
  G4double halfDimY = halfY*nVoxelY;
  G4double halfDimZ = halfZ*nVoxelZ;

  CheckVoxelsFillContainer(halfDimX,
                           halfDimY,
                           halfDimZ);

  //---------------------------------------------------------------------------------------------//
  if (globalInserter!=NULL) {
    if (globalInserter->GetVerbosity()>=1) {
      G4cout << "---- Exiting GateRegularParameterization::BuildRegularParameterization ..."
             << G4endl << std::flush;
    }
  }
  else {
    if (globalFictInserter->GetVerbosity()>=1) {
      G4cout << "---- Exiting GateRegularParameterization::BuildRegularParameterization ..."
             << G4endl << std::flush;
    }
  }
}

G4Material* GateRegularParameterization::ComputeMaterial(const G4int copyNo, G4VPhysicalVolume* pv, const G4VTouchable*)
{
  G4Material*      mp( voxelReader->GetVoxelMaterial(copyNo) );

  // If a physical volume is given (not the case when G4RegularNavigation
  // call ComputeStepSkippingEqualMaterials) we set the visual attributes
  // of the material to the corresponding voxel
  if (pv != 0) {
    G4LogicalVolume* lv( pv->GetLogicalVolume()                  );
    if (translator) lv->SetVisAttributes( translator->GetMaterialAttributes(mp) );
  }

  return mp;
}
