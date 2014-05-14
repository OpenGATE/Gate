/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateImageRegionalizedSubVolumeSolid.hh"
//#include "GateManager.hh"

//====================================================================
/// Builds the solid with its name and the volume which owns it
GateImageRegionalizedSubVolumeSolid::GateImageRegionalizedSubVolumeSolid(const G4String& name, GateImageRegionalizedSubVolume* volume)
  : G4Box(name,
	  volume->GetHalfSize().x(),
	  volume->GetHalfSize().y(),
	  volume->GetHalfSize().z()),
    pVolume(volume)
{
  GateMessage("Volume",5,"GateImageRegionalizedSubVolumeSolid(" << ")" << G4endl);
}
//====================================================================

//====================================================================
/// Destructor
GateImageRegionalizedSubVolumeSolid::~GateImageRegionalizedSubVolumeSolid()
{
  GateMessage("Volume",5,"~GateImageRegionalizedSubVolumeSolid(" << ")" << G4endl);
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
EInside GateImageRegionalizedSubVolumeSolid::Inside(const G4ThreeVector& p) const
{
  //GateDebugMessage("Volume",6,"GateImageRegionalizedSubVolumeSolid["<<GetName()<<"]::Inside("<<p<<")"<<Gateendl);

  return pVolume->Inside(p);
}
//====================================================================


//====================================================================
// Returns the outwards pointing unit normal of the shape for the
// surface closest to the point at offset p.
G4ThreeVector GateImageRegionalizedSubVolumeSolid::SurfaceNormal( const G4ThreeVector& p) const
{
  GateDebugMessage("Volume",6,"GateImageRegionalizedSubVolumeSolid["<<GetName()<<"]::SurfaceNormal("<<p<<")"<<G4endl);
  GateDebugMessage("Volume",6," before correction return " << G4Box::SurfaceNormal(p) << G4endl);
  G4ThreeVector a = pVolume->SurfaceNormal(p);
  //GateError("SurfaceNormal");
  return a;
}
//====================================================================


//====================================================================
// Return the distance along the normalised vector v to the shape,
// from the point at offset p. If there is no intersection, return
// kInfinity. The first intersection resulting from `leaving' a
// surface/volume is discarded. Hence, it is tolerant of points on
// the surface of the shape.
G4double GateImageRegionalizedSubVolumeSolid::DistanceToIn(const G4ThreeVector& p, const G4ThreeVector& v) const
{
  GateDebugMessage("Volume",6,"GateImageRegionalizedSubVolumeSolid["<<GetName()<<"]::DistanceToIn(" <<p<<","<<v<<")"<<G4endl);
  return pVolume->DistanceToIn(p,v);
}
//====================================================================
  

//====================================================================
// Calculate the distance to the nearest surface of a shape from an
// outside point. The distance can be an underestimate.
G4double GateImageRegionalizedSubVolumeSolid::DistanceToIn(const G4ThreeVector& p) const
{
  GateDebugMessage("Volume",6,"GateImageRegionalizedSubVolumeSolid["<<GetName()<<"]::DistanceToIn("<<p<<")"<<G4endl);
  // Distance to bbox of the image
  //  G4double dbox = G4Box::DistanceToIn(p);
  G4double dbox = pVolume->DistanceToIn(p);
  GateDebugMessage("Volume",6," DIST = " << dbox <<G4endl);
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
G4double GateImageRegionalizedSubVolumeSolid::DistanceToOut(const G4ThreeVector& p, 
						  const G4ThreeVector& v,
						  const G4bool calcNorm,
						  G4bool *validNorm, 
						  G4ThreeVector *n) const
{
  // //GateDebugMessage("Volume",6,"GateImageRegionalizedSubVolumeSolid["
  //		   <<GetName()<<"]::DistanceToOut("
  //		   <<p<<","<<v<<","<<calcNorm<<")"<<Gateendl);
 
  return pVolume->DistanceToOut(p,v,calcNorm,validNorm,n);

}
//====================================================================



//====================================================================  
 
// Calculate the distance to the nearest surface of a shape from an
// inside point. The distance can be an underestimate.
G4double GateImageRegionalizedSubVolumeSolid::DistanceToOut(const G4ThreeVector& p) const
{
  return pVolume->DistanceToOut(p);
}
//====================================================================


  
//====================================================================
// Returns a random point located on the surface of the solid.
G4ThreeVector GateImageRegionalizedSubVolumeSolid::GetPointOnSurface() const
{
  //  //GateDebugMessage("Volume",6,"GateImageRegionalizedSubVolumeSolid["<<GetName()<<"]::GetPointOnSurface()"<<Gateendl);
 // GateError("GetPointOnSurface");
  return G4Box::GetPointOnSurface();
}
//====================================================================



//====================================================================
// Provide identification of the class of an object.
// (required for persistency and STEP interface)
G4GeometryType GateImageRegionalizedSubVolumeSolid::GetEntityType() const
{
  return G4String("GateImageRegionalizedSubVolumeSolid");
}
//====================================================================


//====================================================================
// Dumps contents of the solid to a stream.  
std::ostream& GateImageRegionalizedSubVolumeSolid::StreamInfo(std::ostream& os) const
{
  os << "-----------------------------------------------------------\n"
     << "    *** Dump for solid - " << GetName() << " ***\n"
     << "    ===================================================\n"
     << " Solid type: GateImageRegionalizedSubVolumeSolid\n"
     << " Parameters: \n"
     << "    half length X: " << GetXHalfLength()/mm << " mm \n"
     << "    half length Y: " << GetYHalfLength()/mm << " mm \n"
     << "    half length Z: " << GetZHalfLength()/mm << " mm \n"
     << "    label        : " << pVolume->GetLabel() << "\n"
     << "-----------------------------------------------------------\n";

  return os;
}
//====================================================================

  
