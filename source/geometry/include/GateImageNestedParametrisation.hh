/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! 
  \class  GateImageNestedParametrisation : 
  \brief  Messenger of GateImageParametrisedVolume.
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateImageNestedParametrisation__hh__
#define __GateImageNestedParametrisation__hh__

#include "globals.hh"
#include "G4VNestedParameterisation.hh"

#include "GateVImageVolume.hh"
class GateImageNestedParametrisedVolume;

//-----------------------------------------------------------------------------
/// \brief Parametrisation for GateImageParametrisedVolume
class GateImageNestedParametrisation : public G4VNestedParameterisation
{
public:
  //-----------------------------------------------------------------------------
  GateImageNestedParametrisation(GateImageNestedParametrisedVolume* volume);
  ~GateImageNestedParametrisation();  

  //-----------------------------------------------------------------------------
  void ComputeTransformation (const G4int copyNo, G4VPhysicalVolume* physVol) const;

  //-----------------------------------------------------------------------------
 // void ComputeDimensions(G4Box&, const G4int, const G4VPhysicalVolume* ) const;
  
  //-----------------------------------------------------------------------------
  virtual G4Material* ComputeMaterial(G4VPhysicalVolume* currentVol, const G4int repNo, const G4VTouchable* parentTouch=0);

  //-----------------------------------------------------------------------------
  virtual G4Material* ComputeMaterial(const G4int repNo, G4VPhysicalVolume* currentVol, const G4VTouchable* parentTouch) {
	return ComputeMaterial( currentVol, repNo, parentTouch );
  }
  
  //-----------------------------------------------------------------------------
  virtual G4int       GetNumberOfMaterials() const;

  //-----------------------------------------------------------------------------
  virtual G4Material* GetMaterial(G4int idx) const;  

/*
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Tubs &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Trd &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Trap &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Cons &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Sphere &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Orb &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Torus &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Para &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Polycone &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Polyhedra &, const G4int, const G4VPhysicalVolume *) const {}
  
  //-----------------------------------------------------------------------------
  void ComputeDimensions(G4Hype &, const G4int, const G4VPhysicalVolume *) const {}
  */
protected:
  //-----------------------------------------------------------------------------
  /// associated NestedParametrised volume
  GateImageNestedParametrisedVolume* pVolume;
  //-----------------------------------------------------------------------------
  /// vector of label to material correspondance
  std::vector<G4Material*> mVectorLabel2Material;
  //-----------------------------------------------------------------------------
  /// Dummy material (Air)
  G4Material * mAirMaterial;
  //-----------------------------------------------------------------------------
  // Vector of Z positions
  std::vector<G4double>  fpZ;
};
//-----------------------------------------------------------------------------

#endif

