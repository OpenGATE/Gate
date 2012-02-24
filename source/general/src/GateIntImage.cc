/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! \file 
  \brief a 3D image
*/

#ifndef __GateIntImage_cc__
#define __GateIntImage_cc__

#include "GateIntImage.hh"
#include "GateMiscFunctions.hh"

#include "G4ThreeVector.hh"
#include "G4GeometryTolerance.hh"

#ifdef G4ANALYSIS_USE_ROOT
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#endif

//-----------------------------------------------------------------------------
GateIntImage::GateIntImage() {
  halfSize   = G4ThreeVector(0.0, 0.0, 0.0);
  resolution = G4ThreeVector(0.0, 0.0, 0.0);
  mPosition = G4ThreeVector(0.0, 0.0, 0.0);
  UpdateSizesFromResolutionAndHalfSize();
  mOutsideValue = 0;
  kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateIntImage::~GateIntImage() {
   data.clear();

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h) {
  resolution = r;
  halfSize = h;
  UpdateSizesFromResolutionAndHalfSize();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::SetResolutionAndHalfSize(G4ThreeVector r, G4ThreeVector h, G4ThreeVector position) {
  mPosition = position;
  SetResolutionAndHalfSize(r,h);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v) {
  resolution = r;
  voxelSize = v;
  UpdateSizesFromResolutionAndVoxelSize();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::SetResolutionAndVoxelSize(G4ThreeVector r, G4ThreeVector v, G4ThreeVector position) {
  mPosition = position;
  SetResolutionAndVoxelSize(r,v);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::Allocate() {
  UpdateNumberOfValues();
  //GateDebugMessage("IntImage",8,"GateIntImage::Resize " << nbOfValues << Gateendl);
  data.resize(nbOfValues);
  std::fill(data.begin(), data.end(), 0);
  PrintInfo();
  UpdateDataForRootOutput();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateIntImage::GetCoordinatesFromIndex(int index) const{
  return G4ThreeVector (index%planeSize%lineSize, 
			index%planeSize/lineSize, 
			index/planeSize);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateIntImage::GetVoxelCenterFromCoordinates(G4ThreeVector p) const{
  return G4ThreeVector(p.x()*voxelSize.x()+halfSizeMinusVoxelCenter.x(), 
		       p.y()*voxelSize.y()+halfSizeMinusVoxelCenter.y(), 
		       p.z()*voxelSize.z()+halfSizeMinusVoxelCenter.z());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPosition(const G4ThreeVector& position) const{
  //std::cout.precision(20);
  GateDebugMessage("IntImage",9," GetIndex for " << position << G4endl);

  // compute position in voxels (non-integer)
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  GateDebugMessage("Image",9," pos in voxel = " << x << " " << y << " " << z << G4endl);

  // Special case for right borders  :
  if (fabs(x - resolution.x()) <= kCarTolerance*0.5) x -= 2*kCarTolerance;
  if (fabs(y - resolution.y()) <= kCarTolerance*0.5) y -= 2*kCarTolerance;
  if (fabs(z - resolution.z()) <= kCarTolerance*0.5) z -= 2*kCarTolerance;

  // to floor values
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);
  //GateDebugMessage("Image",9," " << ix << " " << iy << " " << iz 
  //				   << " (tol/2.0= " << kCarTolerance*0.5 << ")" << Gateendl);

  // Check boundary : I do not know what to do in this case
  // with 'middle' type : should not append ???
  // if (fabs(ix-x) < kCarTolerance) {
  // 	std::cerr << "Surface x=" << x << " for p=" << position << std::endl;
  //   }
  
  //   if (fabs(iy-y) < kCarTolerance) {
  // 	std::cerr << "Surface y=" << x << " for p=" << position << std::endl;
  //   }
  
  //   if (fabs(iz-z) < kCarTolerance) {
  // 	std::cerr << "Surface z=" << x << " for p=" << position << std::endl;
  //   }
  
  // Check if not out of the volume ... (should not append with 'middle' ?)
  if (ix >= resolution.x()) return -1;
  if (iy >= resolution.y()) return -1;
  if (iz >= resolution.z()) return -1;
  if (ix < 0) return -1;
  if (iy < 0) return -1;
  if (iz < 0) return -1;
  GateDebugMessage("Image",9,ix << " " << iy << " " << iz << G4endl);
  
  return (ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPositionAndDirection(const G4ThreeVector& position,
						const G4ThreeVector& direction) const{
  //std::cout.precision(20);
  // TEMP GateDebugMessage("Image",9," GetIndex for pos " << position << " - dir "<<direction<<Gateendl);

  // compute position in voxels (non-integer)
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  //GateDebugMessage("Image",9," pos in voxel = " << x << " " << y << " " << z << Gateendl);

  // to floor values
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);
  //GateDebugMessage("Image",9," " << ix << " " << iy << " " << iz << Gateendl);

  bool xmin=false, xmax=false;
  bool ymin=false, ymax=false;
  bool zmin=false, zmax=false;  

  //-----------------------------------------------------------------------------
  // Special case for surfaces :
  if ((x - ix < kCarTolerance*0.5/voxelSize.x())&&(direction.x()<0)) {
    ix--;
    xmin=true;
    //GateDebugMessage("Image",9,"GIFPAD XSurf inf : ix = " << ix << Gateendl);
  }
  if ((y - iy < kCarTolerance*0.5/voxelSize.y())&&(direction.y()<0)) {
    iy--;
    ymin=true;
    //GateDebugMessage("Image",9,"GIFPAD YSurf inf : iy = " << iy << Gateendl);
  }
  if ((z - iz < kCarTolerance*0.5/voxelSize.z())&&(direction.z()<0)) {
    iz--;
    zmin=true;
    ///GateDebugMessage("Image",9,"GIFPAD ZSurf inf : iz = " << iz << Gateendl);
  }
  //-----------------------------------------------------------------------------
  
  //-----------------------------------------------------------------------------
  // Special case for corners :
  if (xmin && ymin) {
    //GateDebugMessage("Run",9, "xmin && ymin : " << x << " " << ix << " et " 
    //				 << y << " " << iy << " d=" << direction << Gateendl;);
    if (((x-ix+1)/direction.x()) < ((y-iy+1)/direction.y())) { iy++; ymin = false; }
    else { ix++; xmin = false; }
    //GateDebugMessage("Run",9, "final ix/iy : " << ix << " " << iy << Gateendl;);
  }
  
  if (xmin && zmin) {
    //GateDebugMessage("Run",9, "xmin && zmin : " << x << " " << ix << " et " 
    //			 << z << " " << iz << " d=" << direction << Gateendl);
    if (((x-ix+1)/direction.x()) < ((z-iz+1)/direction.z())) { iz++; zmin = false; }
    else { ix++; xmin = false; }
    //GateDebugMessage("Run",9, "final ix/iz : " << ix << " " << iz << Gateendl;);
  }
  
  if (ymin && zmin) {
    //GateDebugMessage("Run",9, "ymin && zmin : " << y << " " << iy << " et " 
    //				 << z << " " << iz << " d=" << direction << Gateendl);
    if (((y-iy+1)/direction.y()) < ((z-iz+1)/direction.z())) { iz++; zmin = false; }
    else { iy++; ymin = false; }
    //GateDebugMessage("Run",9, "final iy/iz : " << iy << " " << iz << Gateendl;);
  }
  //-----------------------------------------------------------------------------
  
  //-----------------------------------------------------------------------------
  // Special case for surfaces :  
  if ((ix+1 -x < kCarTolerance*0.5/voxelSize.x())&&(direction.x()>0)) {
    ix++;
    xmin=true;
    //GateDebugMessage("Image",9,"GIFPAD XSurf sup : ix = " << ix << Gateendl;);
  }
  if ((iy+1 -y < kCarTolerance*0.5/voxelSize.y())&&(direction.y()>0)) {
    iy++;
    ymin=true;
    //GateDebugMessage("Image",9,"GIFPAD YSurf sup : iy = " << iy << Gateendl;);
  }
  if ((iz+1 -z < kCarTolerance*0.5/voxelSize.z())&&(direction.z()>0)) {
    iz++;
    zmin=true;
    //GateDebugMessage("Image",9,"GIFPAD ZSurf sup : iz = " << iz << Gateendl;);
  }

  //-----------------------------------------------------------------------------
  // Special case for corners :
  if (xmax && ymax) {
    //GateDebugMessage("Run",9, "xmax && ymax : " << x << " " << ix << " et " 
    //				 << y << " " << iy << " d=" << direction << Gateendl;);
    if (((x-ix+1)/direction.x()) < ((y-iy+1)/direction.y())) { iy++; ymax = false; }
    else { ix++; xmax = false; }
    //GateDebugMessage("Run",9, "final ix/iy : " << ix << " " << iy << Gateendl;);
  }
  
  if (xmax && zmax) {
    //GateDebugMessage("Run",9, "xmax && zmax : " << x << " " << ix << " et " 
    //					 << z << " " << iz << " d=" << direction << Gateendl);
    if (((x-ix+1)/direction.x()) < ((z-iz+1)/direction.z())) { iz++; zmax = false; }
    else { ix++; xmax = false; }
    //	GateDebugMessage("Run",9, "final ix/iz : " << ix << " " << iz << Gateendl;);
  }
  
  if (ymax && zmax) {
    //GateDebugMessage("Run",9, "ymax && zmax : " << y << " " << iy << " et " 
    //					 << z << " " << iz << " d=" << direction << Gateendl);
    if (((y-iy+1)/direction.y()) < ((z-iz+1)/direction.z())) { iz++; zmax = false; }
    else { iy++; ymax = false; }
    //GateDebugMessage("Run",9, "final iy/iz : " << iy << " " << iz << Gateendl;);
  }
  //-----------------------------------------------------------------------------

  // Check if not out of the volume ... 
  if (ix >= resolution.x()) return -1;
  if (iy >= resolution.y()) return -1;
  if (iz >= resolution.z()) return -1;
  if (ix < 0) return -1;
  if (iy < 0) return -1;
  if (iz < 0) return -1;
  // TEMP GateMessage("Image",9,ix << " " << iy << " " << iz << Gateendl);
  
  return (ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPostPositionAndDirection(const G4ThreeVector& position,
						    const G4ThreeVector& direction) const{
  //std::cout.precision(20);
  // TEMP GateDebugMessage("Image",9," GetIndex from post for pos " << position << " - dir "<<direction<<Gateendl);

  // compute position in voxels (non-integer)
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  // TEMP GateDebugMessage("Image",9," pos in voxel = " << x << " " << y << " " << z << Gateendl);

  // to floor values
  int ix = (int)floor(x);
  int iy = (int)floor(y);
  int iz = (int)floor(z);

  // TEMP GateDebugMessage("Image",9," " << ix << " " << iy << " " << iz << Gateendl);

  //DS WARNING : kCarTolerance sur 2 ? 
  //DS biais possible dans les sensors ? 

  // Special case for surfaces :
  if ((x - ix < kCarTolerance)&&(direction.x()<0)) {
    ix--;
    // TEMP GateDebugMessage("Image",9,"XSurf inf : ix = " << ix << Gateendl);
  }
  if ((y - iy < kCarTolerance)&&(direction.y()<0)) {
    iy--;
    // TEMP GateDebugMessage("Image",9,"YSurf inf : iy = " << iy << Gateendl);
  }
  if ((z - iz < kCarTolerance)&&(direction.z()<0)) {
    iz--;
    // TEMP GateDebugMessage("Image",9,"ZSurf inf : iz = " << iz << Gateendl);
  }

  if ((ix+1 -x < kCarTolerance)&&(direction.x()>0)) {
    ix++;
    // TEMP GateDebugMessage("Image",9,"XSurf sup : ix = " << ix << Gateendl);
  }
  if ((iy+1 -y < kCarTolerance)&&(direction.y()>0)) {
    iy++;
    // TEMP GateDebugMessage("Image",9,"YSurf sup : iy = " << iy << Gateendl);
  }
  if ((iz+1 -z < kCarTolerance)&&(direction.z()>0)) {
    iz++;
    // TEMP GateDebugMessage("Image",9,"ZSurf sup : iz = " << iz << Gateendl);
  }

  // Check boundary : I do not know what to do in this case
  // with 'middle' type : should not append ???
  // if (fabs(ix-x) < kCarTolerance) {
  // 	std::cerr << "Surface x=" << x << " for p=" << position << std::endl;
  //   }
  
  //   if (fabs(iy-y) < kCarTolerance) {
  // 	std::cerr << "Surface y=" << x << " for p=" << position << std::endl;
  //   }
  
  //   if (fabs(iz-z) < kCarTolerance) {
  // 	std::cerr << "Surface z=" << x << " for p=" << position << std::endl;
  //   }
  
  // Check if not out of the volume ... (should not append with 'middle' ?)
  if (ix >= resolution.x()) return -1;
  if (iy >= resolution.y()) return -1;
  if (iz >= resolution.z()) return -1;
  if (ix < 0) return -1;
  if (iy < 0) return -1;
  if (iz < 0) return -1;
  // TEMP GateMessage("Image",9,ix << " " << iy << " " << iz << Gateendl);
  
  return (ix+iy*lineSize+iz*planeSize);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPostPosition(const G4ThreeVector& pre, 
					const G4ThreeVector& post) const{
  //std::cout.precision(20);
  // TEMP GateDebugMessage("Image",9,"GetIndex pre  : " << pre << Gateendl);
  // TEMP GateDebugMessage("Image",9,"GetIndex post : " << post << Gateendl);

  //DS WARNING : kCarTolerance sur 2 ? 
  //DS biais possible dans les sensors ? 

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


  /*
  // TEMP GateDebugMessage("Image",9,"Position (post) in voxel = " 
  // TEMP 		   << x << " " << y << " " << z 
  // TEMP 		   << Gateendl);

  // check bounds 
  int fx = GetIndexFromPostPosition(x, pre.x(), post.x(), resolution.x());
  int fy = GetIndexFromPostPosition(y, pre.y(), post.y(), resolution.y());
  int fz = GetIndexFromPostPosition(z, pre.z(), post.z(), resolution.z());
  // TEMP GateDebugMessage("Image",9," " << fx << " " << fy << " " << fz << Gateendl);

  if (fx == -1) return -1;
  if (fy == -1) return -1;
  if (fz == -1) return -1;
  */

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPrePosition(const G4ThreeVector& pre, 
				       const G4ThreeVector& post) const{
  //std::cout.precision(20);
  // TEMP GateDebugMessage("Image",9,"GetIndex pre  : " << pre << Gateendl);
  // TEMP GateDebugMessage("Image",9,"GetIndex post : " << post << Gateendl);

  // compute pre voxel coordinates
  double x = (pre.x()+halfSize.x())/voxelSize.x();
  double y = (pre.y()+halfSize.y())/voxelSize.y();
  double z = (pre.z()+halfSize.z())/voxelSize.z();
  // TEMP GateDebugMessage("Image",9,"Position (pre) in voxel = " 
  // TEMP 		   << x << " " << y << " " << z 
  // TEMP 		   << Gateendl);
  // TEMP GateDebugMessage("Image",9,"Position (pre) in voxel = " 
  // TEMP 		   << (int)floor(x) << " " << (int)floor(y) << " " << (int)floor(z) 
  // TEMP 		   << Gateendl);
  
  // DEBUG
  // TEMP double xx = (post.x()+halfSize.x())/voxelSize.x();
  // TEMP double yy = (post.y()+halfSize.y())/voxelSize.y();
  // TEMP double zz = (post.z()+halfSize.z())/voxelSize.z();
  // TEMP GateDebugMessage("Image",9,"[Position (post) in voxel = " 
  // TEMP 		   << xx << " " << yy << " " << zz 
  // TEMP 		   << Gateendl);
  // TEMP GateDebugMessage("Image",9,"[Position (post) in voxel = " 
  // TEMP 				   << (int)floor(xx) << " " << (int)floor(yy) << " " 
  // TEMP 			   << (int)floor(zz) 
  // TEMP 			   << Gateendl);

  // check bounds  
  int fx = GetIndexFromPrePosition(x, pre.x(), post.x(), resolution.x());
  int fy = GetIndexFromPrePosition(y, pre.y(), post.y(), resolution.y());
  int fz = GetIndexFromPrePosition(z, pre.z(), post.z(), resolution.z());
  
  /*
    int fx = (int)floor(x);
    int fy = (int)floor(y);
    int fz = (int)floor(z);
  */
  // TEMP GateDebugMessage("Image",9," " << fx << " " << fy << " " << fz << Gateendl);

  if (fx < 0) return -1;
  if (fy < 0) return -1;
  if (fz < 0) return -1;

  if (fx >= resolution.x()) return -1;
  if (fy >= resolution.y()) return -1;
  if (fz >= resolution.z()) return -1;

  return (fx+fy*lineSize+fz*planeSize);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPostPosition(const double t, 
					const double pret, 
					const double postt, 
					const double resolutiont) const{
  std::cerr << "GetIndexFromPostPosition ! non !" << std::endl;
  exit(0);
  int ft = (int)floor(t);
  int ct = (int)ceil(t);
  // // TEMP GateDebugMessage("Image",9,"kCarTolerance = " << kCarTolerance
  // 				   << "kSurface = " << kSurface << Gateendl);

  /*DS WARNING :
  //DS if move along an edge -> I don't know how to do...
  //DS here stop as a bug ...
  */

  if (fabs(ft - t)<kCarTolerance) { // I am on left edge
    // TEMP GateDebugMessage("Image",9,"Left edge detected" << Gateendl);
    if (pret < postt) {
      // TEMP GateDebugMessage("Image",9,"pre lower : go inside (no change)" << ft << Gateendl);
    }
    else {
      if (pret> postt) {
	ft--;
	// TEMP GateDebugMessage("Image",9,"pre greater : go outside (change)" << ft << Gateendl);
      }
      else {
	//GateError( "Moving along en edge ..." << ft << Gateendl);
	exit(0);
      }
    }
  }
  else {
    if (fabs(ct - t)<kCarTolerance) { // I am on right edge
      // TEMP GateDebugMessage("Image",9,"Right edge detected" << Gateendl);
      if (pret < postt) {
	ft++;
	// TEMP GateDebugMessage("Image",9,"pre t lower : go outside (change) " << ft << Gateendl);
      }
      else {
	if (pret > postt) {
	  // TEMP GateDebugMessage("Image",9,"pre t greater : go inside (no change)" << ft << Gateendl);
	}
	else {
	  //GateError( "Moving along en edge ..." << ft << Gateendl);
	  exit(0);
	}
      }
    }
  }
  
  // check out of volume 
  if (ft<0) {
    // TEMP GateDebugMessage("Image",9,"Outside LEFT." << Gateendl);
    return -1;
  }
  if (ft == resolutiont) {
    // TEMP GateDebugMessage("Image",9,"Outside RIGHT." << Gateendl);
    return -1;
  }
  return ft;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetIndexFromPrePosition(const double t, 
				       const double pret, 
				       const double postt, 
				       const double //resolutiont
				       ) const{
  int ft = (int)floor(t);
  int ct = (int)ceil(t);
  // // TEMP GateDebugMessage("Image",9,"kCarTolerance = " << kCarTolerance
  // 				   << "kSurface = " << kSurface << Gateendl);

  /*DS WARNING :
  //DS if move along an edge -> I don't know how to do...
  //DS here stop as a bug ...
  */

  if (fabs(ft - t)<kCarTolerance) { // I am on left edge
    // TEMP GateDebugMessage("Image",9,"Left edge detected" << Gateendl);
    if (pret < postt) {
      //ft++;
      // TEMP GateDebugMessage("Image",9,"pre lower : go right (no change)" << ft << Gateendl);
    }
    else {
      if (pret> postt) {
	ft--;
	// TEMP GateDebugMessage("Image",9,"pre greater : go left (change)" << ft << Gateendl);
      }
      else {
	//GateError( "Moving along en edge ..." << ft << Gateendl);
	exit(0);
      }
    }
  }
  else {
    if (fabs(ct - t)<kCarTolerance) { // I am on right edge
      // TEMP GateDebugMessage("Image",9,"Right edge detected" << Gateendl);
      // std::cerr << "Right edge detected" << std::endl;
      // exit(0);
      if (pret < postt) {
	ft++;
	// TEMP GateDebugMessage("Image",9,"pre t lower : go right (change) " << ft << Gateendl);
      }
      else {
	if (pret > postt) {
	  // TEMP GateDebugMessage("Image",9,"pre t greater : go left (change)" << ft << Gateendl);
	}
	else {
	  //GateError( "Moving along en edge ..." << ft << Gateendl);
	  exit(0);
	}
      }
    }
  }
  
  // check out of volume // DEVRAIT PAS ETRE NEEDED !!!!!!!!
  // if (ft<0) {
  // 	// TEMP GateDebugMessage("Image",9,"Outside LEFT." << Gateendl);
  // 	return -1;
  //   }
  //   if (ft == resolutiont) {
  // 	// TEMP GateDebugMessage("Image",9,"Outside RIGHT." << Gateendl);
  // 	return -1;
  //   }
  return ft;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::GetCoordinatesFromPosition(const G4ThreeVector & position, int& i, int& j, int& k) {
  // compute position
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  // G4cout << "xyz = " << x << " " << y << " " << z << Gateendl);

  // simple rounding
  i = (int)floor(x);
  j = (int)floor(y);
  k = (int)floor(z);
  // G4cout << ix << " " << iy << " " << iz << " -> ";

  // special case for border
  if (i == resolution.x()) i--;
  if (j == resolution.y()) j--;
  if (k == resolution.z()) k--;
  // G4cout << ix << " " << iy << " " << iz << Gateendl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateIntImage::GetCoordinatesFromPosition(const G4ThreeVector & position) {
  // compute position
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  // TEMP GateDebugMessage("Image",9," xyz = " << x << " " << y << " " << z << Gateendl);

  // simple rounding
  G4ThreeVector i( floor(x), floor(y), floor(z) );

  // TEMP GateDebugMessage("Image",9," floor xyz = " << i << Gateendl);

  // special case for border
  if (i.x() == resolution.x()) i.setX( i.x()-1 );
  if (i.y() == resolution.y()) i.setY( i.y()-1 );
  if (i.z() == resolution.z()) i.setZ( i.z()-1 );
  // TEMP GateDebugMessage("Image",9," floor xyz = " << i << Gateendl);
  
  // no check if is inside
  return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateIntImage::GetNonIntegerCoordinatesFromPosition(G4ThreeVector position) {
  // TEMP GateDebugMessage("Image",9," GetNonIntegerCoordinatesFromPosition position =" 	<< position << " " << Gateendl);

  // compute position
  double x = (position.x()+halfSize.x())/voxelSize.x();
  double y = (position.y()+halfSize.y())/voxelSize.y();
  double z = (position.z()+halfSize.z())/voxelSize.z();
  // TEMP GateDebugMessage("Image",9," xyz = " << x << " " << y << " " << z << Gateendl);

  G4ThreeVector i( x, y, z );
  return i;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateIntImage::GetVoxelCornerFromCoordinates(G4ThreeVector c) const{
  return G4ThreeVector ( c.x() * voxelSize.x() - halfSize.x(), 
			 c.y() * voxelSize.y() - halfSize.y(), 
			 c.z() * voxelSize.z() - halfSize.z() );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::PrintInfo() {
  /*  GateMessage("Image",1,"Matrix Size=\t"      << size        << Gateendl);
      GateMessage("Image",1,"HalfSize=\t"  << halfSize    << Gateendl);
      GateMessage("Image",1,"Resol=\t"     << resolution  << Gateendl);
      GateMessage("Image",1,"VoxelSize=\t" << voxelSize   << Gateendl);
      GateMessage("Image",1,"planeSize=\t" << planeSize   << Gateendl);
      GateMessage("Image",1,"lineSize=\t"  << lineSize    << Gateendl);
      GateMessage("Image",1,"halfSizeMinusVoxelCenter=\t" << halfSizeMinusVoxelCenter << Gateendl);
      GateMessage("Image",1,"nbOfValues=\t"     << nbOfValues  << Gateendl);
      GateMessage("Image",1,"dataSize =\t"     << data.size() << Gateendl);*/
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::UpdateSizesFromResolutionAndHalfSize() {
  // INPUT  : resolution + halfSize
  // OUTPUT : nbOfValues, size, voxelSize, lineSize, planeSize

  //GateMessage("Image",3,"GateImage::UpdateSizesFromResolutionAndHalfSize()"<<Gateendl);

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

  PrintInfo();

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::UpdateSizesFromResolutionAndVoxelSize() {
  // INPUT  : resolution + VoxelSize
  // OUTPUT : nbOfValues, size, HalfSize, lineSize, planeSize

  //GateMessage("Image",3,"GateIntImage::UpdateSizesFromResolutionAndVoxelSize()"<<Gateendl);

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

  PrintInfo();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::UpdateNumberOfValues() {
  nbOfValues = (int)lrint(resolution.x()*resolution.y()*resolution.z());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::Read(G4String filename) {
  //GateMessage("Image",3,"Read GateIntImage " << filename << Gateendl);
  
  G4String extension = getExtension(filename);
  //GateMessage("Image",4,"extension = " << extension << Gateendl);
  
  if (extension == "vox") ReadVox(filename);
  else if (extension == "txt") ReadAscii(filename);
  else if (extension == "hdr") ReadAnalyze(filename);
  else if (extension == "img") ReadAnalyze(filename);
  else if (extension == "img.gz") ReadAnalyze(filename);
  else {
    // GateError( "Unknow image file extension. Knowns extensions are : "
    //	   << G4endl << ".vox, .hdr, .img " << Gateendl);
    exit(0);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::ReadVox(G4String filename) {
  //GateMessage("Image",5,"GateIntImage::ReadVox " << filename << Gateendl);
  // open file  
  std::ifstream is;
  OpenFileInput(filename, is);
  // read header
  std::string s; 
  is >> s;
  if (s != "VOX") {
    // GateError( "Error while opening " << filename 
    //	   << " for reading a Vox : this is not a vox format ?" << Gateendl);
    exit(0);
  }
  is >> s;
  if (s == "v2") ReadVox2(is);
  else {
    if (s == "v3") ReadVox3(is);
    else {
      // GateError( "Error while opening " << filename 
      //     << " for reading a Vox : unsuported vox format." << Gateendl);
      exit(0);
    }
  }
  if (!is) {
    //GateError( "Error while reading " << filename 
    //	   << " header (vox format)" << Gateendl);
    exit(0);
  }
  
  // allocate
  Allocate();
  
  // read data
  //  G4cout << nbOfValues << Gateendl);
  std::vector<unsigned char> temp(nbOfValues);
  is.read((char*)(&(temp[0])), nbOfValues*sizeof(char));
  for(unsigned int i=0; i<temp.size(); i++) {
    data[i] = (int)temp[i];
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::ReadVox2(std::ifstream & is) {
  //GateMessage("Image",5,"GateIntImage::ReadVox2" << Gateendl);
  // read image size
  skipComment(is);
  is >> resolution[0];
  is >> resolution[1];
  is >> resolution[2];
  if (!is) {
    //GateError( "Error while reading image size" << Gateendl);
    exit(0);
  }
  //  G4cout << resolution << Gateendl);
  // read voxel size
  skipComment(is);
  is >> voxelSize[0];
  is >> voxelSize[1];
  is >> voxelSize[2];
  if (!is) {
    //GateError( "Error while reading voxel size" << Gateendl);
    exit(0);
  }
  //  G4cout << voxelSize << Gateendl);
  UpdateSizesFromResolutionAndVoxelSize();
  // read image dimension
  skipComment(is);
  int dim;
  is >> dim;
  if (dim != 3) {
    //GateError( "Error while reading image dim = " << dim 
    //	   <<  " (only dim=3 is allowed)" << Gateendl);
    exit(0);
  }
  //  G4cout << dim << Gateendl);
  // read image type
  std::string valueTypeName;
  skipComment(is);
  is >> valueTypeName;
  if (valueTypeName != "uchar") {
    //GateError( "Error while reading image valueTypeName = " << valueTypeName
    //	   <<  " (only 'uchar' is allowed)" << Gateendl);
    exit(0);
  }
  //GateMessage("Image",8,"Voxel type = " << valueTypeName << Gateendl);
  is.get();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::ReadVox3(std::ifstream & is) {
  // GateMessage("Image",5,"GateIntImage::ReadVox3" << Gateendl);
  // read image dimension
  skipComment(is);
  int dim;
  is >> dim;
  if (dim != 3) {
    //GateError( "Error while reading image dim = " << dim 
    //	   <<  " (only dim=3 is allowed)" << Gateendl);
    exit(0);
  }
  // read point dimension
  is >> dim;
  if (dim != 1) {
    //GateError( "Error while reading point dim = " << dim 
    //	   <<  " (only dim=1 is allowed)" << Gateendl);
    exit(0);
  }
  // read image type
  std::string valueTypeName;
  skipComment(is);
  is >> valueTypeName;
  if (valueTypeName != "uchar") {
    //GateError( "Error while reading image valueTypeName = " << valueTypeName
    //	   <<  " (only 'uchar' is allowed)" << Gateendl);
    exit(0);
  }
  // read image size
  skipComment(is);
  is >> resolution[0];
  is >> resolution[1];
  is >> resolution[2];
  if (!is) {
    //GateError( "Error while reading image size" << Gateendl);
    exit(0);
  }
  // read voxel size
  skipComment(is);
  is >> voxelSize[0];
  is >> voxelSize[1];
  is >> voxelSize[2];
  if (!is) {
    //GateError( "Error while reading voxel size" << Gateendl);
    exit(0);
  }
  UpdateSizesFromResolutionAndVoxelSize();
  // read dim order (not used here)
  skipComment(is);
  is >> dim;
  is >> dim;
  is >> dim;
  is.get();
  is.get();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::ReadAnalyze(G4String filename) {
  //GateMessage("Image",5,"GateIntImage::ReadAnalyze " << filename << Gateendl);
  // Read header
  GateAnalyzeHeader hdr;
  hdr.Read(filename);

  short int rx,ry,rz,rc;
  hdr.GetImageSize(rx,ry,rz,rc);

  PixelType vx,vy,vz;
  hdr.GetVoxelSize(vx,vy,vz);
  		  
  // update sizes and allocate
  resolution = G4ThreeVector(rx,ry,rz);
  voxelSize = G4ThreeVector(vx,vy,vz);
  UpdateSizesFromResolutionAndVoxelSize();
  Allocate();

  // open .img file  
  int l = filename.length();
  filename.replace(l-3,3,"img");
  /*[l-1] = 'g';
    filename[l-2] = 'm';
    filename[l-3] = 'i';
  */
  std::ifstream is;
  OpenFileInput(filename, is);

  // read data
  //GateMessage("Image",8,"nbOfValues = " << nbOfValues << Gateendl);


  // Read values ...
  if (hdr.GetVoxelType() == GateAnalyzeHeader::SignedShortType) {	
    GateMessage("Image",5,"Voxel Type = SignedShortType" << Gateendl);
    typedef short VoxelType;
    std::vector<VoxelType> temp(nbOfValues);
    data.resize(nbOfValues);
    is.read((char*)(&(temp[0])), nbOfValues*sizeof(VoxelType));
    for(unsigned int i=0; i<temp.size(); i++) {
      data[i] = (int)temp[i];
    }
  }
  else if (hdr.GetVoxelType() == GateAnalyzeHeader::FloatType) {
    GateMessage("Image",5,"Voxel Type = FloatType" << Gateendl);
    data.resize(nbOfValues);
    is.read((char*)(&(data[0])), nbOfValues*sizeof(int));
  }
    else if (hdr.GetVoxelType() == GateAnalyzeHeader::SignedIntType) {	
    GateMessage("Image",5,"Voxel Type = SignedIntType" << Gateendl);
    typedef int VoxelType;
    std::vector<VoxelType> temp(nbOfValues);
    data.resize(nbOfValues);
    is.read((char*)(&(temp[0])), nbOfValues*sizeof(VoxelType));
    for(unsigned int i=0; i<temp.size(); i++) {
      data[i] = (int)temp[i];
    }
  }
  else if (hdr.GetVoxelType() == GateAnalyzeHeader::UnsignedCharType) {
    GateMessage("Image",5,"Voxel Type = UnsignedCharType" << Gateendl);
    typedef unsigned char VoxelType;
    std::vector<VoxelType> temp(nbOfValues);
    data.resize(nbOfValues);
    is.read((char*)(&(temp[0])), nbOfValues*sizeof(VoxelType));
    for(unsigned int i=0; i<temp.size(); i++) {
      data[i] = (int)temp[i];
    }
  }
  else {
    GateError("I don't know (yet) this voxel type ... try float or unsigned char");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::ReadAscii(G4String filename) {
  //GateMessage("Image",8,"GateImage::ReadAscii " << filename << Gateendl);

  std::ifstream is;
  OpenFileInput(filename, is);
  
  // Header
  std::string s;
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read ####################################
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read # 
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read Matrix
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read Size=
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read values ...
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read #
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read Resol
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read =
  G4ThreeVector resol;
  is >> resol; 
  //GateDebugMessage("Image",8,"Resol = " << resol << Gateendl);
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read #
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read VoxelSize
  is >> s ;// GateDebugMessage("Image",8,s<< Gateendl); // read =
  G4ThreeVector voxsize;
  is >> voxsize;
  //GateDebugMessage("Image",8,"VoxelSize = " << voxsize << Gateendl);
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read #
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read nbVal
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read =
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read value
  is >> s ; //GateDebugMessage("Image",8,s<< Gateendl); // read ####################################
  
  // set size
  SetResolutionAndVoxelSize(resol, voxsize);
  Allocate();

  // read data
  int dim = 3;
  std::string v;
  if (resolution.x() == 1) dim--;
  if (resolution.y() == 1) dim--;
  if (resolution.z() == 1) dim--;
  //GateDebugMessage("Image",8,"Image dimension is " << dim << Gateendl);
  
  if (dim <= 1) {
    // read values in columns
    for(int i=0; i<nbOfValues; i++) {
      is >> v; //GateDebugMessage("Image",8,"val = " << v << Gateendl); 
      data[i] = atoi(v.c_str()); //GateDebugMessage("Image",8,"val = " << data[i] << Gateendl); 
    }
  }
  if (dim == 2) {
    // write values in line/columns
    double width=0;
    double height=0;
    if (resolution.x() == 1.0) { width = resolution.y(); height = resolution.z(); }
    if (resolution.y() == 1.0) { width = resolution.x(); height = resolution.z(); }
    if (resolution.z() == 1.0) { width = resolution.x(); height = resolution.y(); }
    int i=0;
    for(int y=0; y<height; y++) {
      for(int x=0; x<width; x++) {
	is >> v;
	data[i] = atoi(v.c_str()); 
	i++;
      }
    }
  }
  if (dim == 3) {
    int i=0;
    for(int z=0; z<resolution.z(); z++) {
      for(int y=0; y<resolution.y(); y++) {
	for(int x=0; x<resolution.x(); x++) {
	  is >> v;
	  data[i] = atoi(v.c_str()); 
	  i++;
	}
      }
    }
  }
  if (!is) {
    //GateError( "Error while reading " << Gateendl);
    exit(0);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::MergeDataByAddition(G4String filename) {
  //GateDebugMessage("Image", 5, "GateImage::MergeDataByAddition in " << filename << Gateendl);
  // check if it exist
  std::ifstream is(filename);
  if (!is) {
    //	GateDebugMessage("Image", 5, "Do not exist : do nothing (already merged)" << Gateendl);
    return;
  }
  is.close();
  GateIntImage temp;
  temp.Read(filename);
  GateIntImage::const_iterator pi = temp.begin();
  GateIntImage::const_iterator pe = temp.end();
  GateIntImage::iterator po = begin();
  while (pi != pe) {
    *po = (*po)+(*pi);
    ++po;
    ++pi;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::Write(G4String filename) {
  Write(filename, "");  
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::Write(G4String filename, const G4String & comment) {
  GateMessage("Actor",5,"GateImage::write " << filename << G4endl);  

  G4String extension = getExtension(filename);
  GateMessage("Actor",5,"extension = " << extension << G4endl);

  std::ofstream os;

  if (extension == "bin") {
    // open
    OpenFileOutput(filename, os);
    WriteBin(os);
  }
  else {
    if (extension == "vox") {
      // open
      OpenFileOutput(filename, os);
      WriteVox(os);
    }
    else {
      if (extension == "txt") {
	// open
	GateMessage("Actor",5,"Write text file"<< G4endl);
	OpenFileOutput(filename, os);
	GateMessage("Actor",5,"Write text file"<< G4endl);
	WriteAscii(os, comment);
	GateMessage("Actor",5,"Write text file - end " << G4endl);
      }
      else {
	if (extension == "hdr") {
	  // Header
	  //GateMessage("Image",8,"Analyze format : writing header (" << filename<<")" << Gateendl);
	  GateAnalyzeHeader hdr;
	  hdr.SetVoxelType(GateAnalyzeHeader::FloatType);
	  hdr.SetImageSize((short int)resolution.x(),(short int)resolution.y(),(short int)resolution.z());
	  hdr.SetVoxelSize((PixelType)voxelSize.x(),(PixelType)voxelSize.y(),(PixelType)voxelSize.z());
	  hdr.Write(filename);
	  // Data
	  setExtension(filename,"img");
	  //GateMessage("Image",8,"Analyze format : writing data (" << filename<<")"<< Gateendl);
	  // open
	  OpenFileOutput(filename, os);
	  WriteBin(os);
	}
	else {
	  if (extension == "root") {
	    WriteRoot(filename);
	  }
	  else {
	    //GateMessage("Image",1,"WARNING : Don't know how to write '" << extension 
	    //	   << " format... I try ASCII file" << Gateendl);
	    // open
	    OpenFileOutput(filename, os);
	    WriteAscii(os, comment);
	  }
	}
      }
    }
  }
  os.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::WriteBin(std::ofstream & os) {
  //GateMessage("Image",8,"GateImage::WriteBin " << Gateendl);
  // write
  os.write((char*)(&(data[0])), nbOfValues*sizeof(int));
  if (!os) {
    // GateError( "Error while writing " << Gateendl);
    exit(0);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::WriteVox(std::ofstream & ) {
  // GateMessage("Image",8,"GateIntImage::WriteVox " << Gateendl);
  //GateError("GateIntImage::WriteVox NOT IMPLEMENTED");
  //DS TODO
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::WriteAscii(std::ofstream & os, const G4String & comment) {
  GateMessage("Actor",5,"GateIntImage::WriteAscii " <<G4endl);
  // write comment in header
  os  << "#################################### " << G4endl
      << "# Matrix Size= " << size        << G4endl
      << "# Resol      = " << resolution  << G4endl
      << "# VoxelSize  = " << voxelSize   << G4endl
      << "# nbVal      = " << nbOfValues  << G4endl
      << "#################################### " << G4endl;
  if (comment != "") os << comment << G4endl;
  
  // write data
  int dim = 3;
  if (resolution.x() == 1) dim--;
  if (resolution.y() == 1) dim--;
  if (resolution.z() == 1) dim--;
  GateMessage("Actor",5,"Image dimension is " << dim << G4endl);
  
  if (dim <= 1) {
    // write values in columns
    for(int i=0; i<nbOfValues; i++)
      os << data[i] << std::endl;	
  }
  if (dim == 2) {
    // write values in line/columns
    double width=0;
    double height=0;
    if (resolution.x() == 1.0) { width = resolution.y(); height = resolution.z(); }
    if (resolution.y() == 1.0) { width = resolution.x(); height = resolution.z(); }
    if (resolution.z() == 1.0) { width = resolution.x(); height = resolution.y(); }
    int i=0;
    for(int y=0; y<height; y++) {
      for(int x=0; x<width; x++) {
	os << data[i] << " ";	
	i++;
      }
      os << std::endl;
    }
  }
  if (dim == 3) {
    int i=0;
    for(int z=0; z<resolution.z(); z++) {
      for(int y=0; y<resolution.y(); y++) {
	for(int x=0; x<resolution.x(); x++) {
	  os << data[i] << " ";	
	  i++;
	}
	os << std::endl;
      }
      os << std::endl;
    }
  }
  if (!os) {
    GateError( "WriteAscii: Error while writing ");
    exit(0);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateIntImage::ESide GateIntImage::GetSideFromPointAndCoordinate(const G4ThreeVector & p, const G4ThreeVector & c) {
  GateDebugMessage("Image", 8, "GateIntImage::GetSideFromCoordinate(" << p << "," << c << ")" << Gateendl);
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
int GateIntImage::GetNeighborValueFromCoordinate(const ESide & side, const G4ThreeVector & coord) {
  //GateMessage("Image", 8, "GateIntImage::GetNeighborValueFromCoordinate(" << coord
  //			  << ", side=" << side << Gateendl);
  int ttt;
  G4ThreeVector c(coord);
  switch (side) {
  case kMX:c.setX(coord.x()-1); if (coord.x() <0) ttt=0;  break;
  case kPX:c.setX(coord.x()+1); if (coord.x() >= GetResolution().x()) ttt=0; break;
	
  case kMY:c.setY(coord.y()-1); if (coord.y() <0) ttt=0; break;
  case kPY:c.setY(coord.y()+1); if (coord.y() >= GetResolution().y()) ttt=0; break;

  case kMZ:c.setZ(coord.z()-1); if (coord.z() <0) ttt=0;  break;
  case kPZ:c.setZ(coord.z()+1); if (coord.z() >= GetResolution().z()) ttt=0;  break;

  default: ttt=0;
    //	GateError("I don't know side = " << side);
  }
  
  return GetValue(GetIndexFromCoordinates(c));
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::UpdateDataForRootOutput() {

  // Select data for root output
  mRootHistoDim = 0;
  if (resolution.x() != 1) mRootHistoDim++;
  if (resolution.y() != 1) mRootHistoDim++;
  if (resolution.z() != 1) mRootHistoDim++;

 // std::vector<int> axes(mRootHistoDim);

  if (mRootHistoDim == 1) {
    if (resolution.x() != 1) { 
      mRootHistoBinxNb = resolution.x(); 
      mRootHistoBinxLow = -halfSize.x()+mPosition.x(); 
      mRootHistoBinxUp = halfSize.x()+mPosition.x(); 
    }
    if (resolution.y() != 1) { 
      mRootHistoBinxNb = resolution.y(); 
      mRootHistoBinxLow = -halfSize.y()+mPosition.y(); 
      mRootHistoBinxUp = halfSize.y()+mPosition.y(); 
    }
    if (resolution.z() != 1) { 
      mRootHistoBinxNb = resolution.z(); 
      mRootHistoBinxLow = -halfSize.z()+mPosition.z(); 
      mRootHistoBinxUp = halfSize.z()+mPosition.z(); 
    }

    mRootHistoBinxSize = (mRootHistoBinxUp-mRootHistoBinxLow)/mRootHistoBinxNb;

  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateIntImage::WriteRoot(G4String filename) {
#ifdef G4ANALYSIS_USE_ROOT
  // GateMessage("Image", 0 , "Writing image root output in " << filename << G4endl);

  if (mRootHistoDim == 1) {
    TFile * f = new TFile(filename, "RECREATE");
    TH1F * h = new TH1F("histo", 
			std::string("1D distribution "+filename).c_str(), 
			mRootHistoBinxNb, 
			mRootHistoBinxLow, 
			mRootHistoBinxUp);
    double s = mRootHistoBinxSize/2.0;
    int i=0;
    for(double x=mRootHistoBinxLow+s; x<mRootHistoBinxUp; x+=mRootHistoBinxSize) {
      h->Fill(x, data[i]);
      i++;
    }
    h->Write();
    f->Close();
  }
  else { 
    GateError("sorry try to output 2D or 3D histogram : not yet !!");
  }
#endif
#ifndef G4ANALYSIS_USE_ROOT
  GateError(filename<<" was not created. GATE was compiled without ROOT!");
#endif


}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateIntImage::HasSameResolutionThan(const GateIntImage & image) const {
  if (GetResolution().x() == image.GetResolution().x() && 
      GetResolution().y() == image.GetResolution().y() && 
      GetResolution().z() == image.GetResolution().z()) return true;
  else return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateIntImage::HasSameResolutionThan(const GateIntImage * pImage) const {
    return HasSameResolutionThan(*pImage);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateIntImage::GetMinValue()
{ 
   int minVal = GetValue(0); 
   for(int i =1; i<GetNumberOfValues();i++)
   {
     if(GetValue(i)<minVal) minVal = GetValue(i);
   }
   return minVal; 
} 
//-----------------------------------------------------------------------------


#endif 

