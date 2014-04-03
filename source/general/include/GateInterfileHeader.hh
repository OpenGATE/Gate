/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGeometryVoxelInterfileReader_h
#define GateGeometryVoxelInterfileReader_h 1
#include <stdio.h>
#include <stdlib.h>
#include "globals.hh"
#include "G4ThreeVector.hh"
#include <vector>
#include <map>
 

class GateInterfileHeader
{
public:

  typedef float PixelType;
  GateInterfileHeader();
  ~GateInterfileHeader() {};

  //virtual void Describe(G4int level);

  void ReadHeader(std::string & filename);
  void ReadData(std::string filename, std::vector<float> & data);
  
  void ReadKey(FILE* fp);
  
  
//protected:
  typedef unsigned short G4short;
  
  inline void SwapBytes(G4short* buffer, G4int size) {
  	for (G4int i = 0; i < size; ++i) {
  		buffer[i] = ((buffer[i]>> 8) | (buffer[i] << 8));
  	}
  }
  
  
  G4String m_dataFileName;
  G4int m_numPlanes;
  G4float m_planeThickness;
  G4int m_dim[2];
  G4float m_pixelSize[2];
  G4float m_matrixSize[2];
  G4String m_dataTypeName;
  G4String m_dataByteOrder; 

};

#endif
