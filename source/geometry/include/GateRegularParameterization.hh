/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEREGULARPARAMETERIZATION_HH
#define GATEREGULARPARAMETERIZATION_HH 1

#include "globals.hh"
#include "GateVGeometryVoxelReader.hh"
#include "G4PhantomParameterisation.hh"
#include "G4ThreeVector.hh"

class GateRegularParameterized;
class GateFictitiousVoxelMapParameterized;
class G4VPhysicalVolume;
class G4Material;

class GateRegularParameterization : public G4PhantomParameterisation
{
public:

    //! Constructor
    GateRegularParameterization(GateRegularParameterized* itsInserter,
                                   const G4ThreeVector& voxN );
    GateRegularParameterization(GateFictitiousVoxelMapParameterized* itsInserter,
                                   const G4ThreeVector& voxN );

    //! Destructor
    ~GateRegularParameterization(){;}

    //! Build the regular parameterization
    void BuildRegularParameterization();

    //! Get the total number of voxels
    inline G4int GetNbOfCopies()
      { return static_cast<int>(voxelNumber.x()*voxelNumber.y()*voxelNumber.z()); }

    //! Overload of virtual function ComputeMaterial
    G4Material* ComputeMaterial(const G4int copyNo, G4VPhysicalVolume * aVolume, const G4VTouchable*);

    //! Get the physical volume 'cont_phys' that contains all voxels
    inline G4VPhysicalVolume* GetPhysicalContainer() {return cont_phys;}

private:

  GateRegularParameterized*            globalInserter;
  GateFictitiousVoxelMapParameterized* globalFictInserter;
  GateVGeometryVoxelReader*            voxelReader;
  GateVGeometryVoxelTranslator*        translator;

  const G4ThreeVector voxelSize;
  const G4ThreeVector voxelNumber;

  G4VPhysicalVolume * cont_phys; //! The physical container used by the parameterisation and
                                 //! by the ConstructOwnPhysicalVolumes of ParameterizedInserter
};

#endif
