/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateElectricTabulatedField3D.hh"
#include "GateMessageManager.hh"
#include "GateMiscFunctions.hh"
#include "G4SystemOfUnits.hh"

GateElectricTabulatedField3D::GateElectricTabulatedField3D( G4String filename)  :
   xField(0),yField(0),zField(0),
   nx(0),ny(0),nz(0),
   minx(0), maxx(0), miny(0), maxy(0), minz(0), maxz(0),
   dx(0), dy(0), dz(0),
   invertX(false), invertY(false), invertZ(false),
   lenUnit(cm), fieldUnit(volt/m)
{
    
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "                    Electric field                         " << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "    Reading the field grid from " <<         filename        << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  
  ReadDatabase(filename); // Open the file for reading.
  
  GateMessage("Core", 0, "\n ---> ... done reading " << Gateendl);
  
  
  // G4cout << " Read values of field from file " << filename << endl;
  std::cout << " ---> assumed the order:  x, y, z, Ex, Ey, Ez "
     << "\n ---> Min values x,y,z: "
     << minx/cm << " " << miny/cm << " " << minz/cm << " cm "
     << "\n ---> Max values x,y,z: "
     << maxx/cm << " " << maxy/cm << " " << maxz/cm << " cm " << std::endl;

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
     
GateElectricTabulatedField3D::~GateElectricTabulatedField3D(){

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



void GateElectricTabulatedField3D::ReadDatabase(G4String filename){
    
    std::ifstream file;
	OpenFileInput(filename, file);
	skipComment(file);

	SetDimensions(file);
	skipComment(file);
    
    G4double xval=0.;
    G4double yval=0.;
    G4double zval=0.;
    G4double Ex=0.;
    G4double Ey=0.;
    G4double Ez=0.;
    
    // Read in the data
    
    for (int iz=0; iz<nz; iz++) {
        for (int iy=0; iy<ny; iy++) {
            for (int ix=0; ix<nx; ix++) {
                file >> xval >> yval >> zval >> Ex >> Ey >> Ez;
                if ( ix==0 && iy==0 && iz==0 ) {
                    minx = xval * lenUnit;
                    miny = yval * lenUnit;
                    minz = zval * lenUnit;
                }
                xField[ix][iy][iz] = Ex * fieldUnit;
                yField[ix][iy][iz] = Ey * fieldUnit;
                zField[ix][iy][iz] = Ez * fieldUnit;
            }
        }
    }
    file.close();

    maxx = xval * lenUnit;
    maxy = yval * lenUnit;
    maxz = zval * lenUnit;
}


void GateElectricTabulatedField3D::SetDimensions(std::ifstream & is){
    
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

void GateElectricTabulatedField3D::GetFieldValue(const double point[4],
				      double *Efield ) const
{
    
  double x = point[0];
  double y = point[1];
  double z = point[2];

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
  int xindex = static_cast<int>(std::floor(xdindex));
  int yindex = static_cast<int>(std::floor(ydindex));
  int zindex = static_cast<int>(std::floor(zdindex));

  if ((xindex < 0) || (xindex >= nx - 1) ||
      (yindex < 0) || (yindex >= ny - 1) ||
      (zindex < 0) || (zindex >= nz - 1))
  {
        Efield[0] = 0.0;
        Efield[1] = 0.0;
        Efield[2] = 0.0;
        Efield[3] = 0.0;
        Efield[4] = 0.0;
        Efield[5] = 0.0;
  
  } else{
        // Full 3-dimensional version

        Efield[0] = 0.0;
        Efield[1] = 0.0;
        Efield[2] = 0.0;

        Efield[3] =
          xField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          xField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          xField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          xField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          xField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          xField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          xField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          xField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal ;
        Efield[4] =
          yField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          yField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          yField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          yField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          yField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          yField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          yField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          yField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal ;
        Efield[5] =
          zField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          zField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          zField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          zField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          zField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          zField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          zField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          zField[xindex+1][yindex+1][zindex+1] *   xlocal  *    ylocal  *    zlocal ;
  }
}
