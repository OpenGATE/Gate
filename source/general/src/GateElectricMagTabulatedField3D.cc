/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateElectricMagTabulatedField3D.hh"
#include "GateMessageManager.hh"
#include "GateMiscFunctions.hh"
#include "G4SystemOfUnits.hh"

GateElectricMagTabulatedField3D::GateElectricMagTabulatedField3D( G4String filename)  :
   xEField(0),yEField(0),zEField(0), 
   xBField(0), yBField(0), zBField(0), 
   nx(0),ny(0),nz(0),
   minx(0), maxx(0), miny(0), maxy(0), minz(0), maxz(0),
   dx(0), dy(0), dz(0),
   invertX(false), invertY(false), invertZ(false),
   lenUnit_E(cm), fieldUnit_E(volt/m), 
   lenUnit_B(cm), fieldUnit_B(tesla)
{
    
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "                    Electromagnetic field                  " << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "    Reading the field grid from " <<         filename        << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  GateMessage("Core", 0, "-----------------------------------------------------------" << Gateendl);
  
  ReadDatabase(filename); // Open the file for reading.
  
  GateMessage("Core", 0, "\n ---> ... done reading " << Gateendl);
  
  
  // G4cout << " Read values of field from file " << filename << endl;
  std::cout << " ---> assumed the order:  x, y, z, Ex, Ey, Ez, Bx, By, Bz "
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
     
GateElectricMagTabulatedField3D::~GateElectricMagTabulatedField3D(){

	for (int ix=0; ix<nx; ix++) {
		for (int iy=0; iy<ny; iy++) {
			xEField[ix][iy].clear();
			yEField[ix][iy].clear();
			zEField[ix][iy].clear();
      xBField[ix][iy].clear();
			yBField[ix][iy].clear();
			zBField[ix][iy].clear();
		}
	}
    for (int ix=0; ix<nx; ix++) {
		xEField[ix].clear();
		yEField[ix].clear();
		zEField[ix].clear();
    xBField[ix].clear();
		yBField[ix].clear();
		zBField[ix].clear();
	}
	xEField.clear();
	yEField.clear();
	zEField.clear();
  xBField.clear();
	yBField.clear();
	zBField.clear();
}



void GateElectricMagTabulatedField3D::ReadDatabase(G4String filename){
    
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
    G4double Bx=0.;
    G4double By=0.;
    G4double Bz=0.;
    
    // Read in the data
    
    for (int iz=0; iz<nz; iz++) {
        for (int iy=0; iy<ny; iy++) {
            for (int ix=0; ix<nx; ix++) {
                file >> xval >> yval >> zval >> Ex >> Ey >> Ez >> Bx >> By >> Bz;
                if ( ix==0 && iy==0 && iz==0 ) {
                    minx = xval * lenUnit_E;
                    miny = yval * lenUnit_E;
                    minz = zval * lenUnit_E;
                }
                xEField[ix][iy][iz] = Ex * fieldUnit_E;
                yEField[ix][iy][iz] = Ey * fieldUnit_E;
                zEField[ix][iy][iz] = Ez * fieldUnit_E;
                xBField[ix][iy][iz] = Bx * fieldUnit_B;
                yBField[ix][iy][iz] = By * fieldUnit_B;
                zBField[ix][iy][iz] = Bz * fieldUnit_B;
            }
        }
    }
    file.close();

    maxx = xval * lenUnit_E;
    maxy = yval * lenUnit_E;
    maxz = zval * lenUnit_E;
}


void GateElectricMagTabulatedField3D::SetDimensions(std::ifstream & is){
    
    // Read table dimensions
    is >> nx >> ny >> nz; // Note dodgy order

    std::cout << "  [ Number of values x,y,z: "
	 << nx << " " << ny << " " << nz << " ] "
	 << std::endl;
     
    // Set up storage space for table
    xEField.resize( nx );
    yEField.resize( nx );
    zEField.resize( nx );
    xBField.resize( nx );
    yBField.resize( nx );
    zBField.resize( nx );
    
    for (int ix=0; ix<nx; ix++) {
        xEField[ix].resize(ny);
        yEField[ix].resize(ny);
        zEField[ix].resize(ny);
        xBField[ix].resize(ny);
        yBField[ix].resize(ny);
        zBField[ix].resize(ny);
            for (int iy=0; iy<ny; iy++) {
                xEField[ix][iy].resize(nz);
                yEField[ix][iy].resize(nz);
                zEField[ix][iy].resize(nz);
                xBField[ix][iy].resize(nz);
                yBField[ix][iy].resize(nz);
                zBField[ix][iy].resize(nz);
            }
    }

}

void GateElectricMagTabulatedField3D::GetFieldValue(const double point[4],
				      double *EBfield ) const
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
        EBfield[0] = 0.0;
        EBfield[1] = 0.0;
        EBfield[2] = 0.0;
        EBfield[3] = 0.0;
        EBfield[4] = 0.0;
        EBfield[5] = 0.0;
  
  } else{
        // Full 3-dimensional version

        EBfield[0] =
          xBField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          xBField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          xBField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          xBField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          xBField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          xBField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          xBField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          xBField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal ;
        EBfield[1] =
          yBField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          yBField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          yBField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          yBField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          yBField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          yBField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          yBField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          yBField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal ;
        EBfield[2] =
          zBField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          zBField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          zBField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          zBField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          zBField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          zBField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          zBField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          zBField[xindex+1][yindex+1][zindex+1] *   xlocal  *    ylocal  *    zlocal ;
        EBfield[3] =
          xEField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          xEField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          xEField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          xEField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          xEField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          xEField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          xEField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          xEField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal ;
        EBfield[4] =
          yEField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          yEField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          yEField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          yEField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          yEField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          yEField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          yEField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          yEField[xindex+1][yindex+1][zindex+1] *    xlocal  *    ylocal  *    zlocal ;
        EBfield[5] =
          zEField[xindex  ][yindex  ][zindex  ] * (1-xlocal) * (1-ylocal) * (1-zlocal) +
          zEField[xindex  ][yindex  ][zindex+1] * (1-xlocal) * (1-ylocal) *    zlocal  +
          zEField[xindex  ][yindex+1][zindex  ] * (1-xlocal) *    ylocal  * (1-zlocal) +
          zEField[xindex  ][yindex+1][zindex+1] * (1-xlocal) *    ylocal  *    zlocal  +
          zEField[xindex+1][yindex  ][zindex  ] *    xlocal  * (1-ylocal) * (1-zlocal) +
          zEField[xindex+1][yindex  ][zindex+1] *    xlocal  * (1-ylocal) *    zlocal  +
          zEField[xindex+1][yindex+1][zindex  ] *    xlocal  *    ylocal  * (1-zlocal) +
          zEField[xindex+1][yindex+1][zindex+1] *   xlocal  *    ylocal  *    zlocal ;
  }
}
