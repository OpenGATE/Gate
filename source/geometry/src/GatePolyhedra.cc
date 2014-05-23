/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"

#include "GatePolyhedra.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

G4double GatePolyhedra::DistanceToOut( const G4ThreeVector &p, 
                                       const G4ThreeVector &v,
                                       const G4bool calcNorm,
                                             G4bool *validNorm,
                                             G4ThreeVector *n ) const
{
  G4double distance;
  G4ThreeVector v2(v);
  G4int i=0;
 
  while ( ( distance = G4Polyhedra::DistanceToOut(p,v2,calcNorm,validNorm,n) ) == kInfinity )
  {
  	v2.rotateZ(0.001*degree);
	i++;
  }

  // Uncomment the line below to check the hexagon-trap bug for your system
  // if (i) G4cout << G4endl << "Particle in a polyhedra can't find the way out: trajectory rotated by " << i << " millidegrees" << G4endl << G4endl;

  return distance;
}
