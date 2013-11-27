/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class GateImageRegularParametrisation : used by GateImageRegularParametrisedVolume
*/

#ifndef __GATEIMAGEREGULARPARAMETRISEDVOLUME__HH__
#define __GATEIMAGEREGULARPARAMETRISEDVOLUME__HH__

#include "globals.hh"
#include "G4PhantomParameterisation.hh"
#include "GateVImageVolume.hh"

class GateImageRegularParametrisedVolume;

//-----------------------------------------------------------------------------
/// \brief Parametrisation for GateImageParametrisedVolume
class GateImageRegularParametrisation : public G4PhantomParameterisation
{
public:

  // Constructor
  GateImageRegularParametrisation(GateImageRegularParametrisedVolume* volume);

  // Destructor
  ~GateImageRegularParametrisation();

  // Build the parametrisation
  void BuildRegularParameterisation(G4LogicalVolume * l);

  // inherited from G4PhantomParameterisation
  virtual G4Material* ComputeMaterial(const G4int repNo, G4VPhysicalVolume* currentVol, const G4VTouchable* parentTouch);

protected:

  // associated NestedParametrised volume
  GateImageRegularParametrisedVolume* pVolume;

  // vector of label to material correspondance
  std::vector<G4Material*> mVectorLabel2Material;

  // Dummy material (Air)
  G4Material * mAirMaterial;

  // Vector of Z positions
  std::vector<G4double>  fpZ;
};
//-----------------------------------------------------------------------------

#endif
