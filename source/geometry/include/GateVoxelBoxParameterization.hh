/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVoxelBoxParameterization_H
#define GateVoxelBoxParameterization_H 1

#include "globals.hh"
#include "GateVGeometryVoxelReader.hh"
#include "GatePVParameterisation.hh"
#include "G4ThreeVector.hh"
#include "GateUtilityForG4ThreeVector.hh"
#include "G4Box.hh"
#include "G4VTouchable.hh"

class G4VPhysicalVolume;
class G4Material;

class GateVoxelBoxParameterization : public GatePVParameterisation
{ 
  public:
  
  //! Constructor.
  GateVoxelBoxParameterization( GateVGeometryVoxelReader* voxR, const G4ThreeVector& voxN, const G4ThreeVector& voxS);

  ~GateVoxelBoxParameterization(){;}
 
 
  using G4VPVParameterisation::ComputeMaterial;
  G4Material* ComputeMaterial(const G4int copyNo, G4VPhysicalVolume * aVolume);


  void ComputeTransformation(const G4int copyNo,  G4VPhysicalVolume *aVolume) const;
  using G4VPVParameterisation::ComputeDimensions;
  void ComputeDimensions(G4Box*, const G4int ,const G4VPhysicalVolume* ) const;

  //! Implementation of the pure virtual method declared by the base class GatePVParameterization
  //! This method returns the total number of voxels in the matrix
  inline int GetNbOfCopies() { return static_cast<int>(voxelNumber.x()*voxelNumber.y()*voxelNumber.z()); }

private:

  //  Calculate the three (array) indices of the current (copyNo) voxel
  inline G4ThreeVector ComputeArrayIndices(G4int copyNo)const{

    div_t  qr  ( div(copyNo, nxy) );
    div_t  qr2 ( div(qr.rem, nx ) );
    
    return G4ThreeVector( qr2.rem, qr2.quot, qr.quot );
    
  }

  
private:
  const G4ThreeVector voxelNumber;
  const G4ThreeVector voxelSize;
  const G4ThreeVector voxelZeroPos;

  GateVGeometryVoxelReader* voxelReader;
  GateVGeometryVoxelTranslator* translator;
  
  const int nxy;
  const int nx;
  
};

#endif


