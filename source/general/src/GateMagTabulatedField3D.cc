/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateMagTabulatedField3D.hh"
#include "GateMessageManager.hh"
#include "GateMiscFunctions.hh"
#include "G4SystemOfUnits.hh"

GateMagTabulatedField3D::GateMagTabulatedField3D(G4String filename) :
   xField(0),yField(0),zField(0),
   nx(0),ny(0),nz(0),
   minx(0), maxx(0), miny(0), maxy(0), minz(0), maxz(0),
   dx(0), dy(0), dz(0),
   invertX(false), invertY(false), invertZ(false),
   lenUnit(cm), fieldUnit(tesla)
{    

  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "                    Magnetic field                         " << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "    Reading the field grid from " <<         filename        << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);

  ReadDatabase(filename); // Open the file for reading.

  GateMessage("Core", 0, "\n ---> ... done reading " << Gateendl);


  // G4cout << " Read values of field from file " << filename << endl; 
  std::cout << " ---> assumed the order:  x, y, z, Bx, By, Bz "
	 << "\n ---> Min values x,y,z: " 
	 << minx/cm << " " << miny/cm << " " << minz/cm << " cm "
	 << "\n ---> Max values x,y,z: " 
	 << maxx/cm << " " << maxy/cm << " " << maxz/cm << " cm "<< std::endl;

  // Should really check that the limits are not the wrong way around.
  if (maxx < minx) {std::swap(maxx,minx); invertX = true;}
  if (maxy < miny) {std::swap(maxy,miny); invertY = true;}
  if (maxz < minz) {std::swap(maxz,minz); invertZ = true;}

  std::cout << "\nAfter reordering if neccesary"
	 << "\n ---> Min values x,y,z: " 
	 << minx/cm << " " << miny/cm << " " << minz/cm << " cm "
	 << " \n ---> Max values x,y,z: " 
	 << maxx/cm << " " << maxy/cm << " " << maxz/cm << " cm ";

  dx = maxx - minx;
  dy = maxy - miny;
  dz = maxz - minz;

  GateMessage("Core", 0, "\n ---> Dif values x,y,z (range): "
	 << dx/cm << " " << dy/cm << " " << dz/cm << " cm in z "
	 << "\n-----------------------------------------------------------" << Gateendl);
}

GateMagTabulatedField3D::~GateMagTabulatedField3D(){

	for (int ix=0; ix<nx; ix++) {
		for (int iy=0; iy<ny; iy++) {
			xField[ix][iy].clear();
			yField[ix][iy].clear();
			zField[ix][iy].clear();
		}
	}
    for (int ix=0; ix<nx; ix++) {
		xField[ix].clear();
		yField[ix].clear();
		zField[ix].clear();
	}
	xField.clear();
	yField.clear();
	zField.clear();
}

void GateMagTabulatedField3D::ReadDatabase(G4String filename){

	std::ifstream file;
	OpenFileInput(filename, file);
	skipComment(file);

	SetDimensions(file);
	skipComment(file);

	double xval,yval,zval,bx,by,bz;
	double permeability;

	// Read in the data

	for (int ix=0; ix<nx; ix++) {
		for (int iy=0; iy<ny; iy++) {
			for (int iz=0; iz<nz; iz++) {
				file >> xval >> yval >> zval >> bx >> by >> bz >> permeability;
				if ( ix==0 && iy==0 && iz==0 ) {
					minx = xval * lenUnit;
					miny = yval * lenUnit;
					minz = zval * lenUnit;
				}
				xField[ix][iy][iz] = bx * fieldUnit;
				yField[ix][iy][iz] = by * fieldUnit;
				zField[ix][iy][iz] = bz * fieldUnit;
			}
		}
	}
	file.close();

	maxx = xval * lenUnit;
	maxy = yval * lenUnit;
	maxz = zval * lenUnit;
}

void GateMagTabulatedField3D::SetDimensions(std::ifstream & is){

	// Read table dimensions
	is >> nx >> ny >> nz; // Note dodgy order

	std::cout << "  [ Number of values x,y,z: "
	 << nx << " " << ny << " " << nz << " ] "
	 << std::endl;

	// Set up storage space for table
	xField.resize( nx );
	yField.resize( nx );
	zField.resize( nx );

	for (int ix=0; ix<nx; ix++) {
		xField[ix].resize(ny);
		yField[ix].resize(ny);
		zField[ix].resize(ny);
			for (int iy=0; iy<ny; iy++) {
				xField[ix][iy].resize(nz);
				yField[ix][iy].resize(nz);
				zField[ix][iy].resize(nz);
			}
	}

}

void GateMagTabulatedField3D::GetFieldValue(const double point[4],
				      double *Bfield ) const
{

  double x = point[0];
  double y = point[1];
  double z = point[2];

  // Check that the point is within the defined region 
  if ( x>=minx && x<=maxx &&
       y>=miny && y<=maxy && 
       z>=minz && z<=maxz ) {
    
    // Position of given point within region, normalized to the range
    // [0,1]
    double xfraction = (x - minx) / dx;
    double yfraction = (y - miny) / dy; 
    double zfraction = (z - minz) / dz;

    if (invertX) { xfraction = 1 - xfraction;}
    if (invertY) { yfraction = 1 - yfraction;}
    if (invertZ) { zfraction = 1 - zfraction;}

    // Need addresses of these to pass to modf below.
    // modf uses its second argument as an OUTPUT argument.
    double xdindex, ydindex, zdindex;
    
    // Position of the point within the cuboid defined by the
    // nearest surrounding tabulated points
    double xlocal = ( std::modf(xfraction*(nx-1), &xdindex));
    double ylocal = ( std::modf(yfraction*(ny-1), &ydindex));
    double zlocal = ( std::modf(zfraction*(nz-1), &zdindex));
    
    // The indices of the nearest tabulated point whose coordinates
    // are all less than those of the given point
    int xindex = static_cast<int>(xdindex);
    int yindex = static_cast<int>(ydindex);
    int zindex = static_cast<int>(zdindex);

        // Full 3-dimensional version
	Bfield[0] =
	 (xField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
	  xField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
	  xField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
	  xField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
	  xField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
	  xField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
	  xField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
	  xField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal);

	Bfield[1] =
	 (yField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
	  yField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
	  yField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
	  yField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
	  yField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
	  yField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
	  yField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
	  yField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal);

	Bfield[2] =
	 (zField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
	  zField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
	  zField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
	  zField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
	  zField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
	  zField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
	  zField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
	  zField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal);

  } else {
    Bfield[0] = 0.0;
    Bfield[1] = 0.0;
    Bfield[2] = 0.0;
  }
}

