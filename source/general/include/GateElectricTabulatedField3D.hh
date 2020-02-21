/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "globals.hh"
#include "G4ElectricField.hh"
#include "G4ElectroMagneticField.hh"
#include "G4ios.hh"

#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

class GateElectricTabulatedField3D: public G4ElectricField
{
public:
  GateElectricTabulatedField3D(G4String filename);
  virtual ~GateElectricTabulatedField3D();
  void  GetFieldValue( const  double Point[4], double *Efield) const;
  
private:
  void  ReadDatabase(G4String filename);
  void  SetDimensions(std::ifstream & is);
  
  // Storage space for the table
  vector< vector< vector< double > > > xField;
  vector< vector< vector< double > > > yField;
  vector< vector< vector< double > > > zField;
  // The dimensions of the table
  int nx,ny,nz; 
  // The physical limits of the defined region
  double minx, maxx, miny, maxy, minz, maxz;
  // The physical extent of the defined region
  double dx, dy, dz;
  bool invertX, invertY, invertZ;
  double lenUnit;
  double fieldUnit;
};

