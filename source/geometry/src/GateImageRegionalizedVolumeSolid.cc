/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateImageRegionalizedVolumeSolid.hh"
#include "GateImageRegionalizedVolume.hh"
//#include "GateManager.hh"

//====================================================================
/// Builds the solid with its name and the volume which owns it
GateImageRegionalizedVolumeSolid::GateImageRegionalizedVolumeSolid(const G4String& name, GateImageRegionalizedVolume* volume)
  : G4Box(name,
	  volume->GetHalfSize().x(),
	  volume->GetHalfSize().y(),
	  volume->GetHalfSize().z()),
    GateImageBox(*volume->GetImage(), name),
    mVolume(volume)
{
  ////GateMessage("Volume",5,"GateImageRegionalizedVolumeSolid()" << Gateendl);
}
//====================================================================

//====================================================================
/// Destructor
GateImageRegionalizedVolumeSolid::~GateImageRegionalizedVolumeSolid()
{
  ////GateMessage("Volume",5,"~GateImageRegionalizedVolumeSolid()" << Gateendl);
}
//====================================================================

//====================================================================
// Returns kOutside if the point at offset p is outside the shapes
// boundaries plus Tolerance/2, kSurface if the point is <= Tolerance/2
// from a surface, otherwise kInside.
// Uses G4Box::Inside to know if the points falls in the image 
// and if so picks the label of image to compare it to the label of Gate solid
// *** WARNING *** 
// The current implementation of the kSurface tolerance assumes that the 
// voxel's size if larger than kCarTolerance 
// (which is consistent with the fact that G4Box of smaller size than  
// kCarTolerance cannot be created)
EInside GateImageRegionalizedVolumeSolid::Inside(const G4ThreeVector& p) const
{
  GateDebugMessage("Volume",6,"\t\tGateImageRegionalizedVolumeSolid["<<GetName()<<"]::Inside("<<p<<")"<<G4endl);

  // If is outside of the bounding box, is outside
  EInside bi = G4Box::Inside(p);
  if (bi==kOutside) {
    GateDebugMessage("Volume",6,"\t\t*** OUTSIDE ***"<<G4endl);
  }
  else if (bi==kInside) {
    GateDebugMessage("Volume",6,"\t\t*** INSIDE ***"<<G4endl);
  }
  else if (bi==kSurface) {
    GateDebugMessage("Volume",6,"\t\t*** SURFACE ***"<<G4endl);
  }
  return bi;
}
//====================================================================



//====================================================================
// Return the distance along the normalised vector v to the shape,
// from the point at offset p. If there is no intersection, return
// kInfinity. The first intersection resulting from `leaving' a
// surface/volume is discarded. Hence, it is tolerant of points on
// the surface of the shape.
G4double GateImageRegionalizedVolumeSolid::DistanceToIn(const G4ThreeVector& p, 
					    const G4ThreeVector& v) const
{
  GateDebugMessage("Volume",7,"GateImageRegionalizedVolumeSolid["<<GetName()
		   <<"]::DistanceToIn("<<p<<","<<v<<")"<<G4endl);

  // Distance to bbox of the image
  G4double dbox = G4Box::DistanceToIn(p,v);
  GateDebugMessage("Volume",7," DIST(BB) = " << dbox <<G4endl);
  return dbox;
}
//====================================================================
  

//====================================================================
// Calculate the distance to the nearest surface of a shape from an
// outside point. The distance can be an underestimate.
G4double GateImageRegionalizedVolumeSolid::DistanceToIn(const G4ThreeVector& p) const
{
  GateDebugMessage("Volume",7,"GateImageRegionalizedVolumeSolid["<<GetName()
		   <<"]::DistanceToIn("<<p<<")"<<G4endl);
  // Distance to bbox of the image
  G4double dbox = G4Box::DistanceToIn(p);
  GateDebugMessage("Volume",7," DIST(BB) = " << dbox <<G4endl);
  return dbox;
}
//====================================================================  


//====================================================================
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
G4double GateImageRegionalizedVolumeSolid::DistanceToOut(const G4ThreeVector& p, 
						  const G4ThreeVector& v,
						  const G4bool calcNorm,
						  G4bool *validNorm, 
						  G4ThreeVector *n) const
{
  GateDebugMessage("Volume",7,"GateImageRegionalizedVolumeSolid["<<GetName()
		   <<"]::DistanceToOut("
		   <<p<<","<<v<<","<<calcNorm<<")"<<G4endl);
  // Distance to bbox of the image
  G4double dbox = G4Box::DistanceToOut(p,v,calcNorm,validNorm,n);
  GateDebugMessage("Volume",7," DIST = " << dbox <<G4endl);
  return dbox;
}
//====================================================================


//====================================================================  
 
// Calculate the distance to the nearest surface of a shape from an
// inside point. The distance can be an underestimate.
G4double GateImageRegionalizedVolumeSolid::DistanceToOut(const G4ThreeVector& p) const
{
  GateDebugMessage("Volume",7,"GateImageRegionalizedVolumeSolid["<<GetName()<<"]::DistanceToOut("<<p<<")"<<G4endl);
  // Distance to bbox of the image
  G4double dbox = G4Box::DistanceToOut(p);
  GateDebugMessage("Volume",7," DIST = " << dbox <<G4endl);
  return dbox;
}
//====================================================================


 //====================================================================
// Provide identification of the class of an object.
// (required for persistency and STEP interface)
G4GeometryType GateImageRegionalizedVolumeSolid::GetEntityType() const
{
  return G4String("GateImageRegionalizedVolumeSolid");
}
//====================================================================


//====================================================================
// Dumps contents of the solid to a stream.  
std::ostream& GateImageRegionalizedVolumeSolid::StreamInfo(std::ostream& os) const
{
  os << "-----------------------------------------------------------\n"
     << "    *** Dump for solid - " << GetName() << " ***\n"
     << "    ===================================================\n"
     << " Solid type: GateImageRegionalizedVolumeSolid\n"
     << " Parameters: \n"
     << "    half length X: " << GetXHalfLength()/mm << " mm \n"
     << "    half length Y: " << GetYHalfLength()/mm << " mm \n"
     << "    half length Z: " << GetZHalfLength()/mm << " mm \n"
     << "-----------------------------------------------------------\n";

  return os;
}
//====================================================================

  
