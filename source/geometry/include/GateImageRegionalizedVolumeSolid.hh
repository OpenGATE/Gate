/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
   \class GateImageRegionalizedVolumeSolid
   \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr 
*/

#ifndef __GateImageRegionalizedVolumeSolid__hh__
#define __GateImageRegionalizedVolumeSolid__hh__

#include "GateImageBox.hh"

class GateImageRegionalizedVolume;

//====================================================================
///  \brief  A G4VSolid descendent of G4Box which represents the bounding box of a GateImageRegionalizedVolume.
class GateImageRegionalizedVolumeSolid : virtual public GateImageBox
{
public:
  //====================================================================
  /// Builds the solid with its name and the volume which owns it
  GateImageRegionalizedVolumeSolid(const G4String& name,
				   GateImageRegionalizedVolume* volume);
  /// Destructor
  virtual ~GateImageRegionalizedVolumeSolid();
  //====================================================================

  EInside Inside(const G4ThreeVector& p) const;
  // Returns kOutside if the point at offset p is outside the shapes
  // boundaries plus Tolerance/2, kSurface if the point is <= Tolerance/2
  // from a surface, otherwise kInside.
  // Uses G4Box::Inside to know if the points falls in the image 
  // and if so picks the label of image to compare it to the label of Gate solid

  //  G4ThreeVector SurfaceNormal( const G4ThreeVector& p) const;
  // Returns the outwards pointing unit normal of the shape for the
  // surface closest to the point at offset p.
  
  G4double DistanceToIn(const G4ThreeVector& p, const G4ThreeVector& v) const;
  // Return the distance along the normalised vector v to the shape,
  // from the point at offset p. If there is no intersection, return
  // kInfinity. The first intersection resulting from `leaving' a
  // surface/volume is discarded. Hence, it is tolerant of points on
  // the surface of the shape.
  
  G4double DistanceToIn(const G4ThreeVector& p) const;
  // Calculate the distance to the nearest surface of a shape from an
  // outside point. The distance can be an underestimate.
  // G4Box::DistanceToIn() is ok as can be an underestimate.
  
  G4double DistanceToOut(const G4ThreeVector& p, const G4ThreeVector& v,
			 const G4bool calcNorm=false,
			 G4bool *validNorm=0, G4ThreeVector *n=0) const;
  // Return the distance along the normalised vector v to the shape,
  // from a point at an offset p inside or on the surface of the shape.
  // Intersections with surfaces, when the point is < Tolerance/2 from a
  // surface must be ignored.
  // If calcNorm==true:
  //    validNorm set true if the solid lies entirely behind or on the
  //              exiting surface.
  //    n set to exiting outwards normal vector (undefined Magnitude).
  //    validNorm set to false if the solid does not lie entirely behind
  //              or on the exiting surface
  // If calcNorm==false:
  //    validNorm and n are unused.
  //
  // Must be called as solid.DistanceToOut(p,v) or by specifying all
  // the parameters.
  
  G4double DistanceToOut(const G4ThreeVector& p) const;
  // Calculate the distance to the nearest surface of a shape from an
  // inside point. The distance can be an underestimate.

  G4GeometryType GetEntityType() const;
  // Provide identification of the class of an object.
  // (required for persistency and STEP interface)
  
  // G4ThreeVector GetPointOnSurface() const; 
  // Returns a random point located on the surface of the solid.

  std::ostream& StreamInfo(std::ostream& os) const;
  // Dumps contents of the solid to a stream.

  
  // Functions for visualization
  /* TODAY : inherited from G4Box
  void          DescribeYourselfTo (G4VGraphicsScene& scene) const;
  // A "double dispatch" function which identifies the solid
  // to the graphics scene.
  G4VisExtent   GetExtent          () const;
  // Provide extent (bounding box) as possible hint to the graphics view.
  G4Polyhedron* CreatePolyhedron   () const;
  G4NURBS*      CreateNURBS        () const;
  // Create a G4Polyhedron/G4NURBS/...  (It is the caller's responsibility
  // to delete it).  A null pointer means "not created".
  */

protected:
  /// The volume of which it is the solid
  GateImageRegionalizedVolume* mVolume;

};
// EO class GateImageRegionalizedVolumeSolid
//====================================================================

// EOF
#endif
