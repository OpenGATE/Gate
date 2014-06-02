/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePolyhedra_h
#define GatePolyhedra_h 1

#include "G4Polyhedra.hh"

/*! \class  GatePolyhedra
    \brief  Class derived from G4Polyhedra as a quick-fix to the hexagon-trapping bug
    
    - GatePolyhedra - by Daniel.Strul@iphe.unil.ch 
    
    - This class overloads G4VCSGfaceted::DistanceToOut() to fix the problem caused
      by particules trapped in hexagons
*/      
class GatePolyhedra : public G4Polyhedra
{
  public:

  //! Constructor: calls the G4Polyhedra's constructor
  GatePolyhedra( const G4String& name, 
                       G4double phiStart,    // initial phi starting angle
                       G4double phiTotal,    // total phi angle
                       G4int numSide,        // number sides
                       G4int numZPlanes,     // number of z planes
                 const G4double zPlane[],    // position of z planes
                 const G4double rInner[],    // tangent distance to inner surface
                 const G4double rOuter[]  ) // tangent distance to outer surface
	: G4Polyhedra(name,phiStart,phiTotal,numSide,numZPlanes,zPlane,rInner,rOuter)
	{}



  //! Constructor: calls the G4Polyhedra's constructor
  GatePolyhedra( const G4String& name, 
                       G4double phiStart,    // initial phi starting angle
                       G4double phiTotal,    // total phi angle
                       G4int    numSide,     // number sides
                       G4int    numRZ,       // number corners in r,z space
                 const G4double r[],         // r coordinate of these corners
                 const G4double z[]       )  // z coordinate of these corners
	: G4Polyhedra(name,phiStart,phiTotal,numSide,numRZ,r,z)
	{}
	
  //! Overload of G4VCSGfaceted::DistanceToOut to fix the hexagon-trapping bug
  using G4VCSGfaceted::DistanceToOut;
  virtual G4double DistanceToOut( const G4ThreeVector& p,
                                  const G4ThreeVector& v,
                                  const G4bool calcNorm=false,
                                        G4bool *validNorm=0,
                                        G4ThreeVector *n=0 ) const;
};

#endif
