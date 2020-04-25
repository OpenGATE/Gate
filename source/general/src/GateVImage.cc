/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*! \file
  \brief a 3D image
*/

#ifndef __GATEVIMAGE_CC__
#define __GATEVIMAGE_CC__

// g4
#include "G4ThreeVector.hh"
#include "G4GeometryTolerance.hh"

// std
#include <iomanip>

// gate
#include "GateVImage.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"
#include "GateMHDImage.hh"
#include "GateInterfileHeader.hh"

// root
#ifdef G4ANALYSIS_USE_ROOT
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#endif

//-----------------------------------------------------------------------------
GateVImage::GateVImage() {
  halfSize   = G4ThreeVector(0.0, 0.0, 0.0);
  resolution = G4ThreeVector(0.0, 0.0, 0.0);
  mPosition = G4ThreeVector(0.0, 0.0, 0.0);
  origin = G4ThreeVector(0.0, 0.0, 0.0);
  UpdateSizesFromResolutionAndHalfSize();
  kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVImage::~GateVImage() {
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImage::SetResolutionAndHalfSizeCylinder(G4ThreeVector r, G4ThreeVector h, G4ThreeVector position) {
  mPosition = position;
   SetResolutionAndHalfSizeCylinder(r,h);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImage::SetResolutionAndHalfSizeCylinder(G4ThreeVector r, G4ThreeVector h) {
  resolution = r;
  halfSize = h;
  UpdateSizesFromResolutionAndHalfSizeCylinder();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVImage::SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h) {
  resolution = r;
  halfSize = h;
  UpdateSizesFromResolutionAndHalfSize();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h, G4ThreeVector position) {
  mPosition = position;
  SetResolutionAndHalfSize(r,h);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v) {
  resolution = r;
  voxelSize = v;
  UpdateSizesFromResolutionAndVoxelSize();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v, G4ThreeVector position) {
  mPosition = position;
  SetResolutionAndVoxelSize(r,v);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4ThreeVector GateVImage::GetCoordinatesFromIndex(int index) const{
  return G4ThreeVector (index%planeSize%lineSize,
			index%planeSize/lineSize,
			index/planeSize);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4ThreeVector GateVImage::GetVoxelCenterFromCoordinates(G4ThreeVector p) const{
  return G4ThreeVector(p.x()*voxelSize.x()+halfSizeMinusVoxelCenter.x(),
		       p.y()*voxelSize.y()+halfSizeMinusVoxelCenter.y(),
		       p.z()*voxelSize.z()+halfSizeMinusVoxelCenter.z());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateVImage::GetIndexFromPosition(const G4ThreeVector& position) const{
  //std::cout.precision(20);
  GateDebugMessage("Image",9," GetIndex for " << position << Gateendl);

  // compute position in voxels (non-integer)
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  GateDebugMessage("Image",9," pos in voxel = " << x << " " << y << " " << z << Gateendl);

  // Special case for right borders  :
  if (fabs(x - resolution.x()) <= kCarTolerance*0.5) x -= 2*kCarTolerance;
  if (fabs(y - resolution.y()) <= kCarTolerance*0.5) y -= 2*kCarTolerance;
  if (fabs(z - resolution.z()) <= kCarTolerance*0.5) z -= 2*kCarTolerance;

  // to floor values
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);

  // Check if not out of the volume ... (should not append with 'middle' ?)
  if (ix >= resolution.x()) return -1;
  if (iy >= resolution.y()) return -1;
  if (iz >= resolution.z()) return -1;
  if (ix < 0) return -1;
  if (iy < 0) return -1;
  if (iz < 0) return -1;
  GateDebugMessage("Image",9,ix << " " << iy << " " << iz << Gateendl);

  return (ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateVImage::GetIndexFromPositionCylindricalCS(const G4ThreeVector& position) const{
  //std::cout.precision(20);
  GateDebugMessage("Image",9," GetIndex for " << position << Gateendl);
    //===============================
  //=== new coordinate system
  //=== x1^2 + x2^2 = r^2
  //=== (x1, x2, h)
  //===============================
   G4ThreeVector myV;
  ////beam is comming from x direction
  if (resolution.z()==1)
  {
	  
	  	double rSq = position.z()*position.z() + position.y()*position.y();
		
		// calculating the square root of rSq using Babylonian Method
		// equal to std::sqrt(rSq)
		double r1 = rSq / 2;
		double r2 = r1+(rSq-r1)/2;
		double abortCriterion = voxelSize.y()/200;
		while (std::abs(r2 - r1) > abortCriterion) {
			 r2 = (r1 + r2) / 2;
			 r1 = rSq / r2;
		}
	   // ------------------- end sqrt
	   
	  // compute position in voxels (non-integer)
	  double y = (r1)/voxelSize.y();
	  double x = (position.x()+halfSize.x())/voxelSize.x();
	  GateDebugMessage("Image",9," pos in voxel = " << x << " " << y << " " << z << Gateendl);
	
	  // Special case for right borders  :
	  if (fabs(x - resolution.x()) <= kCarTolerance*0.5) x -= 2*kCarTolerance;
	  if (fabs(y - resolution.y()) <= kCarTolerance*0.5) y -= 2*kCarTolerance;
	   
	  // to floor values
	  int ix = (int)floor(x);
	  int iy = (int)floor(y);

	  // Check if not out of the volume ... (should not append with 'middle' ?)
	  if (iy*2 >= resolution.y()) return -1; //  /2 because radius is only half of diameter
	  if (ix >= resolution.x()) return -1;
	  if (ix < 0) return -1;
	  if (iy < 0) return -1;
	  GateDebugMessage("Image",9,ix << " " << iy << " " << iz << Gateendl);
	  return (ix+iy*lineSize);
  }
  
  ////beam is comming from z direction
  else if (resolution.y()==1)
  {
	 
		  double rSq = position.x()*position.x() + position.y()*position.y();
		
		// calculating the square root of rSq using Babylonian Method
		// equal to std::sqrt(rSq)
		double r1 = rSq / 2;
		double r2 = r1+(rSq-r1)/2;
		double abortCriterion = voxelSize.x()/200;
		while (std::abs(r2 - r1) > abortCriterion) {
			 r2 = (r1 + r2) / 2;
			 r1 = rSq / r2;
		}
	   // ------------------- end sqrt
	   
	  // compute position in voxels (non-integer)
	  double x = (r1)/voxelSize.x();
	  double z = (position.z()+halfSize.z())/voxelSize.z();
	  GateDebugMessage("Image",9," pos in voxel = " << x << " " << y << " " << z << Gateendl);
	
	  // Special case for right borders  :
	  if (fabs(x - resolution.x()) <= kCarTolerance*0.5) x -= 2*kCarTolerance;
	  if (fabs(z - resolution.z()) <= kCarTolerance*0.5) z -= 2*kCarTolerance;
	   
	  // to floor values
	  int ix = (int)floor(x);
	  int iz = (int)floor(z);

		
	  // Check if not out of the volume ... (should not append with 'middle' ?)
	  if (ix*2 >= resolution.x()) return -1; //  /2 because radius is only half of diameter
	  if (iz >= resolution.z()) return -1;
	  if (ix < 0) return -1;
	  if (iz < 0) return -1;
	  GateDebugMessage("Image",9,ix << " " << iy << " " << iz << Gateendl);
	  return (ix+iz*planeSize);
	  
  }
  else
  {
	  return -1;
  }
}


//-----------------------------------------------------------------------------
int GateVImage::GetIndexFromPositionAndDirection(const G4ThreeVector& position,
						const G4ThreeVector& direction) const{
   // compute position in voxels (non-integer)
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();

  // to floor values
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);

  bool xmin=false, xmax=false;
  bool ymin=false, ymax=false;
  bool zmin=false, zmax=false;

  //-----------------------------------------------------------------------------
  // Special case for surfaces :
  if ((x - ix < kCarTolerance*0.5/voxelSize.x())&&(direction.x()<0)) {
    ix--;
    xmin=true;
  }
  if ((y - iy < kCarTolerance*0.5/voxelSize.y())&&(direction.y()<0)) {
    iy--;
    ymin=true;
  }
  if ((z - iz < kCarTolerance*0.5/voxelSize.z())&&(direction.z()<0)) {
    iz--;
    zmin=true;
  }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  // Special case for corners :
  if (xmin && ymin) {
    if (((x-ix+1)/direction.x()) < ((y-iy+1)/direction.y())) { iy++; ymin = false; }
    else { ix++; xmin = false; }
  }

  if (xmin && zmin) {
    if (((x-ix+1)/direction.x()) < ((z-iz+1)/direction.z())) { iz++; zmin = false; }
    else { ix++; xmin = false; }
  }

  if (ymin && zmin) {
    if (((y-iy+1)/direction.y()) < ((z-iz+1)/direction.z())) { iz++; zmin = false; }
    else { iy++; ymin = false; }
  }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  // Special case for surfaces :
  if ((ix+1 -x < kCarTolerance*0.5/voxelSize.x())&&(direction.x()>0)) {
    ix++;
    xmin=true;
  }
  if ((iy+1 -y < kCarTolerance*0.5/voxelSize.y())&&(direction.y()>0)) {
    iy++;
    ymin=true;
  }
  if ((iz+1 -z < kCarTolerance*0.5/voxelSize.z())&&(direction.z()>0)) {
    iz++;
    zmin=true;
  }

  //-----------------------------------------------------------------------------
  // Special case for corners :
  if (xmax && ymax) {
    if (((x-ix+1)/direction.x()) < ((y-iy+1)/direction.y())) { iy++; ymax = false; }
    else { ix++; xmax = false; }
  }

  if (xmax && zmax) {
    if (((x-ix+1)/direction.x()) < ((z-iz+1)/direction.z())) { iz++; zmax = false; }
    else { ix++; xmax = false; }
  }

  if (ymax && zmax) {
    if (((y-iy+1)/direction.y()) < ((z-iz+1)/direction.z())) { iz++; zmax = false; }
    else { iy++; ymax = false; }
  }
  //-----------------------------------------------------------------------------

  // Check if not out of the volume ...
  if (ix >= resolution.x()) return -1;
  if (iy >= resolution.y()) return -1;
  if (iz >= resolution.z()) return -1;
  if (ix < 0) return -1;
  if (iy < 0) return -1;
  if (iz < 0) return -1;

  return (ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateVImage::GetIndexFromPostPositionAndDirection(const G4ThreeVector& position,
						    const G4ThreeVector& direction) const{
  // compute position in voxels (non-integer)
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();

  // to floor values
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);

  // Special case for surfaces :
  if ((x - ix < kCarTolerance)&&(direction.x()<0)) {
    ix--;
  }
  if ((y - iy < kCarTolerance)&&(direction.y()<0)) {
    iy--;
  }
  if ((z - iz < kCarTolerance)&&(direction.z()<0)) {
    iz--;
  }

  if ((ix+1 -x < kCarTolerance)&&(direction.x()>0)) {
    ix++;
  }
  if ((iy+1 -y < kCarTolerance)&&(direction.y()>0)) {
    iy++;
  }
  if ((iz+1 -z < kCarTolerance)&&(direction.z()>0)) {
    iz++;
  }

  // Check boundary : I do not know what to do in this case
  // with 'middle' type : should not append ???
  // if (fabs(ix-x) < kCarTolerance) {
  // 	std::cerr << "Surface x=" << x << " for p=" << position << Gateendl;
  //   }

  //   if (fabs(iy-y) < kCarTolerance) {
  // 	std::cerr << "Surface y=" << x << " for p=" << position << Gateendl;
  //   }

  //   if (fabs(iz-z) < kCarTolerance) {
  // 	std::cerr << "Surface z=" << x << " for p=" << position << Gateendl;
  //   }

  // Check if not out of the volume ... (should not append with 'middle' ?)
  if (ix >= resolution.x()) return -1;
  if (iy >= resolution.y()) return -1;
  if (iz >= resolution.z()) return -1;
  if (ix < 0) return -1;
  if (iy < 0) return -1;
  if (iz < 0) return -1;

  return (ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateVImage::GetIndexFromPostPosition(const G4ThreeVector& pre,
					const G4ThreeVector& post) const{
   // compute post voxel coordinates
  double x = (post.x()+halfSize.x())/voxelSize.x();
  double y = (post.y()+halfSize.y())/voxelSize.y();
  double z = (post.z()+halfSize.z())/voxelSize.z();

  double ix = floor(x);
  double iy = floor(y);
  double iz = floor(z);

  // displacement vector
  double dx = post.x() - pre.x();
  double dy = post.y() - pre.y();
  double dz = post.z() - pre.z();

  bool xm = ( (dx<0) && ((x-ix)<kCarTolerance) );
  bool xp = ( (dx>0) && ((ix+1-x)<kCarTolerance) );
  bool ym = ( (dy<0) && ((y-iy)<kCarTolerance) );
  bool yp = ( (dy>0) && ((iy+1-y)<kCarTolerance) );
  bool zm = ( (dz<0) && ((z-iz)<kCarTolerance) );
  bool zp = ( (dz>0) && ((iz+1-z)<kCarTolerance) );

  double tx(1e30),ty(1e30),tz(1e30);
  if (xm) tx = (ix-x) / dx;
  else if (xp) tx = (ix+1-x) / dx;
  if (ym) ty = (iy-y) / dy;
  else if (yp) ty = (iy+1-y) / dy;
  if (zm) tz = (iz-z) / dz;
  else if (zp) tz = (iz+1-z) / dz;

  if ( xm || xp || ym || yp || zm || zp ) {
    if (tx < ty) {
      if (tx < tz) {
	if (xm) ix--;
	else ix++;
      }
      else {
	if (zm) iz--;
	else iz++;
      }
    }
    else {
      if (ty < tz) {
	if (ym) iy--;
	else iy++;
      }
      else {
	if (zm) iz--;
	else iz++;
      }
    }
  }
  if (ix == -1) return -1;
  if (iy == -1) return -1;
  if (iz == -1) return -1;

  return (int)(ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::GetCoordinatesFromPosition(const G4ThreeVector & position, int& i, int& j, int& k) {
  // compute position
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();

  // simple rounding
  i = (int)floor(x);
  j = (int)floor(y);
  k = (int)floor(z);

  // special case for border
  if (i == resolution.x()) i--;
  if (j == resolution.y()) j--;
  if (k == resolution.z()) k--;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4ThreeVector GateVImage::GetCoordinatesFromPosition(const G4ThreeVector & position) {
  // compute position
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();

  // simple rounding
  G4ThreeVector i( floor(x), floor(y), floor(z) );

  // special case for border
  if (i.x() == resolution.x()) i.setX( i.x()-1 );
  if (i.y() == resolution.y()) i.setY( i.y()-1 );
  if (i.z() == resolution.z()) i.setZ( i.z()-1 );

  // no check if is inside
  return i;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4ThreeVector GateVImage::GetNonIntegerCoordinatesFromPosition(G4ThreeVector position) {
  // compute position
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  G4ThreeVector i( x, y, z );
  return i;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4ThreeVector GateVImage::GetVoxelCornerFromCoordinates(G4ThreeVector c) const{
  return G4ThreeVector ( c.x() * voxelSize.x() - halfSize.x(),
			 c.y() * voxelSize.y() - halfSize.y(),
			 c.z() * voxelSize.z() - halfSize.z() );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::UpdateSizesFromResolutionAndHalfSize() {
  // INPUT  : resolution + halfSize
  // OUTPUT : nbOfValues, size, voxelSize, lineSize, planeSize
  nbOfValues = (int)lrint(resolution.x()*resolution.y()*resolution.z());
  size = G4ThreeVector(halfSize.x()*2.0,
		       halfSize.y()*2.0,
		       halfSize.z()*2.0);
  voxelSize = G4ThreeVector(size.x()/resolution.x(),
			    size.y()/resolution.y(),
			    size.z()/resolution.z());
  voxelVolume = voxelSize.x()*voxelSize.y()*voxelSize.z();
  halfSizeMinusVoxelCenter =
    G4ThreeVector(-halfSize.x()+voxelSize.x()/2.0,
		  -halfSize.y()+voxelSize.y()/2.0,
		  -halfSize.z()+voxelSize.z()/2.0);
  lineSize = (int)lrint(resolution.x());
  planeSize = (int)lrint(resolution.x()*resolution.y());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::UpdateSizesFromResolutionAndHalfSizeCylinder() {
  // INPUT  : resolution + halfSize
  // OUTPUT : nbOfValues, size, voxelSize, lineSize, planeSize
  nbOfValues = (int)lrint(resolution.x()*resolution.y()*resolution.z());
  size = G4ThreeVector(halfSize.x()*2.0,
		       halfSize.y()*2.0,
		       halfSize.z()*2.0);
  voxelSize = G4ThreeVector(size.x()/resolution.x(),
			    size.y()/resolution.y(),
			    size.z()/resolution.z());
  voxelVolume = voxelSize.x()*voxelSize.y()*voxelSize.z();
  halfSizeMinusVoxelCenter =
    G4ThreeVector(-halfSize.x()+voxelSize.x()/2.0,
		  -halfSize.y()+voxelSize.y()/2.0,
		  -halfSize.z()+voxelSize.z()/2.0);
  lineSize = (int)lrint(resolution.x());
  planeSize = (int)lrint(resolution.x()*resolution.y());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::UpdateSizesFromResolutionAndVoxelSize() {
  // INPUT  : resolution + VoxelSize
  // OUTPUT : nbOfValues, size, HalfSize, lineSize, planeSize
  nbOfValues = (int)lrint(resolution.x()*resolution.y()*resolution.z());

  size = G4ThreeVector ( resolution.x() * voxelSize.x(),
			 resolution.y() * voxelSize.y(),
			 resolution.z() * voxelSize.z() );
  halfSize = G4ThreeVector(size.x() / 2.0,
			   size.y() / 2.0,
			   size.z() / 2.0);

  voxelVolume = voxelSize.x()*voxelSize.y()*voxelSize.z();
  halfSizeMinusVoxelCenter =
    G4ThreeVector(-halfSize.x()+voxelSize.x()/2.0,
		  -halfSize.y()+voxelSize.y()/2.0,
		  -halfSize.z()+voxelSize.z()/2.0);
  lineSize = (int)lrint(resolution.x());
  planeSize = (int)lrint(resolution.x()*resolution.y());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::UpdateNumberOfValues() {
  nbOfValues = (int)lrint(resolution.x()*resolution.y()*resolution.z());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVImage::ESide GateVImage::GetSideFromPointAndCoordinate(const G4ThreeVector & p, const G4ThreeVector & c) {
  GateDebugMessage("Image", 8, "GateVImage::GetSideFromCoordinate(" << p << "," << c << ")\n");
  const G4ThreeVector & v = GetVoxelCornerFromCoordinates(c);
  const G4ThreeVector & s = GetVoxelSize();

  GateDebugMessage("Image", 8, "Voxel = " << v << Gateendl);
  GateDebugMessage("Image", 8, "Spaci = " << s << Gateendl);
  GateDebugMessage("Image", 8, "kCarTolerance*0.5 = " << kCarTolerance*0.5 << Gateendl);

  GateDebugMessage("Image", 8, "XL  = " << p.x()-v.x() << Gateendl);
  GateDebugMessage("Image", 8, "XR  = " << v.x()+s.x()-p.x() << Gateendl);
  GateDebugMessage("Image", 8, "YL  = " << p.y()-v.y() << Gateendl);
  GateDebugMessage("Image", 8, "YR  = " << v.y()+s.y()-p.y() << Gateendl);
  GateDebugMessage("Image", 8, "ZL  = " << p.z()-v.z() << Gateendl);
  GateDebugMessage("Image", 8, "ZR  = " << v.z()+s.z()-p.z() << Gateendl);

  if (p.x()-v.x() <= kCarTolerance*0.5) return kMX; // left side
  if (v.x()+s.x()-p.x() <= kCarTolerance*0.5) return kPX; // right side

  if (p.y()-v.y() <= kCarTolerance*0.5) return kMY; // up
  if (v.y()+s.y()-p.y() <= kCarTolerance*0.5) return kPY; // down

  if (p.z()-v.z() <= kCarTolerance*0.5) return kMZ; // front
  if (v.z()+s.z()-p.z() <= kCarTolerance*0.5) return kPZ; // rear

  return kUndefined;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVImage::UpdateDataForRootOutput() {

  // Select data for root output
  mRootHistoDim = 0;
  if (resolution.x() != 1) mRootHistoDim++;
  if (resolution.y() != 1) mRootHistoDim++;
  if (resolution.z() != 1) mRootHistoDim++;

  if (mRootHistoDim == 1 || mRootHistoDim == 2 || mRootHistoDim == 3) {
    if (resolution.x() != 1) {
      mRootHistoBinxNb = resolution.x();
      mRootHistoBinxLow = -halfSize.x()+mPosition.x();
      mRootHistoBinxUp = halfSize.x()+mPosition.x();
    }
    else if (resolution.y() != 1) {
      mRootHistoBinxNb = resolution.y();
      mRootHistoBinxLow = -halfSize.y()+mPosition.y();
      mRootHistoBinxUp = halfSize.y()+mPosition.y();
    }
    else if (resolution.z() != 1) {
      mRootHistoBinxNb = resolution.z();
      mRootHistoBinxLow = -halfSize.z()+mPosition.z();
      mRootHistoBinxUp = halfSize.z()+mPosition.z();
    }

    mRootHistoBinxSize = (mRootHistoBinxUp-mRootHistoBinxLow)/mRootHistoBinxNb;

  }
  if (mRootHistoDim == 2 || mRootHistoDim == 3) {
    /*if (resolution.x() != 1) {
      mRootHistoBinxNb = resolution.x();
      mRootHistoBinxLow = -halfSize.x()+mPosition.x();
      mRootHistoBinxUp = halfSize.x()+mPosition.x();
      }*/
    if (resolution.y() != 1) {
      mRootHistoBinyNb = resolution.y();
      mRootHistoBinyLow = -halfSize.y()+mPosition.y();
      mRootHistoBinyUp = halfSize.y()+mPosition.y();
    }
    else if (resolution.z() != 1) {
      mRootHistoBinyNb = resolution.z();
      mRootHistoBinyLow = -halfSize.z()+mPosition.z();
      mRootHistoBinyUp = halfSize.z()+mPosition.z();
    }

    mRootHistoBinySize = (mRootHistoBinyUp-mRootHistoBinyLow)/mRootHistoBinyNb;

  }
    if (mRootHistoDim == 3) {

      mRootHistoBinzNb = resolution.z();
      mRootHistoBinzLow = -halfSize.z()+mPosition.z();
      mRootHistoBinzUp = halfSize.z()+mPosition.z();


    mRootHistoBinzSize = (mRootHistoBinzUp-mRootHistoBinzLow)/mRootHistoBinzNb;

  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateVImage::HasSameResolutionThan(const GateVImage & image) const {
  if (GetResolution().x() == image.GetResolution().x() &&
      GetResolution().y() == image.GetResolution().y() &&
      GetResolution().z() == image.GetResolution().z()) return true;
  else return false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateVImage::HasSameResolutionThan(const GateVImage * pImage) const {
    return HasSameResolutionThan(*pImage);
}
//-----------------------------------------------------------------------------

#endif
