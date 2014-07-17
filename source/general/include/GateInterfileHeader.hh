/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEINTERFILEREADER_HH
#define GATEINTERFILEREADER_HH 1

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include "globals.hh"
#include "G4ThreeVector.hh"

//-----------------------------------------------------------------------------
class GateInterfileHeader
{
public:

  typedef float PixelType;
  GateInterfileHeader();
  ~GateInterfileHeader() {};

  void ReadHeader(G4String headerFileName);
  void ReadData(std::vector<PixelType> & data);
  void ReadData(G4String dataFileName, std::vector<PixelType> & data);

  void ReadKey(FILE* fp);

  G4String m_headerFileName;
  G4String m_dataFileName;
  G4int m_numPlanes;
  G4float m_planeThickness;
  G4int m_dim[2];
  G4float m_pixelSize[2];
  G4float m_matrixSize[2];
  G4String m_dataTypeName;
  G4int m_bytePerPixel;
  G4int m_dataByteOrder;
  G4int m_offset;
private:
  template <class VoxelType> void DoDataRead(std::vector<PixelType> &data);
  G4bool m_isHeaderInfoRead;
};
//-----------------------------------------------------------------------------

#endif
