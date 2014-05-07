/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateCompressedVoxelParameterization_H
#define GateCompressedVoxelParameterization_H 1

#include "globals.hh"
#include "GateVGeometryVoxelReader.hh"
#include "GatePVParameterisation.hh"
#include "G4ThreeVector.hh"
#include "G4Box.hh"
#include "GateUtilityForG4ThreeVector.hh"

class G4VPhysicalVolume;
class G4Material;

//---------------------------------------------------------------------------
class GateCompressedVoxelParameterization : public GatePVParameterisation
{ 
public:
  
  //! Constructor.
  GateCompressedVoxelParameterization( GateVGeometryVoxelReader* voxR, const G4ThreeVector& voxN, const G4ThreeVector& voxS);

  ~GateCompressedVoxelParameterization(){;}


  virtual G4Material* ComputeMaterial      (const G4int copyNo, G4VPhysicalVolume * aVolume, const G4VTouchable*);
    
  
  using G4VPVParameterisation::ComputeDimensions;
   void ComputeDimensions(G4Box &,
				 const G4int,
				 const G4VPhysicalVolume *) const;
   void ComputeTransformation(const G4int copyNo,  G4VPhysicalVolume *aVolume) const;
    
  //! Implementation of the pure virtual method declared by the base class GatePVParameterization
  int GetNbOfCopies(); 


private:

  
private:
  const G4ThreeVector           voxelNumber;
  const G4ThreeVector           voxelSize;
  const G4ThreeVector           voxelZeroPos;

  GateVGeometryVoxelReader*     voxelReader;
  GateVGeometryVoxelTranslator* voxelTranslator;

  const int nxy;
  const int nx;
  
};

#endif


