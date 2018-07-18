/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEINTERFILEREADER_HH
#define GATEINTERFILEREADER_HH 1

// std
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>

// g4
#include "globals.hh"
#include "G4ThreeVector.hh"

// gate
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"

//-----------------------------------------------------------------------------
class GateInterfileHeader
{
public:

  typedef float DefaultPixelType;
  GateInterfileHeader();
  ~GateInterfileHeader() {};

  void ReadHeader(G4String headerFileName);

  template<class VoxelType>
  void ReadData(std::vector<VoxelType> & data);

  template<class VoxelType>
  void ReadData(G4String dataFileName, std::vector<VoxelType> & data);

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
  template <class ReadPixelType, class OutputPixelType>
  void DoDataRead(std::vector<OutputPixelType> &data);
  G4bool m_isHeaderInfoRead;
};
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
template <class ReadPixelType, class OutputPixelType>
void GateInterfileHeader::DoDataRead(std::vector<OutputPixelType> &data) {
  G4int pixelNumber = m_dim[0]*m_dim[1]*m_numPlanes ;
  std::ifstream is;
  OpenFileInput(m_dataFileName, is);
  std::vector<ReadPixelType> temp(pixelNumber);

  data.resize(pixelNumber);
  is.seekg(m_offset, std::ios::beg);
  is.read((char*)(&(temp[0])), pixelNumber*sizeof(ReadPixelType));
  if (!is) {
    G4cerr << Gateendl <<"Error: the number of pixels that were read from the data file (" << is.gcount() << ") \n"
           << "is inferior to the number computed from its header file (" << pixelNumber << ")!\n";
    G4Exception( "GateInterfileHeader.cc InterfileTooShort", "InterfileTooShort", FatalException, "Correct problem then try again... Sorry!" );
  }
  for(unsigned int i=0; i<temp.size(); i++) {
    if ( BYTE_ORDER != m_dataByteOrder ) {
      ReadPixelType t = temp[i];
      GateMachine::SwapEndians( t );
      temp[i] = t;
      //	  GateMachine::SwapEndians( temp[i] );
    }
    data[i] = (OutputPixelType)temp[i];
  }
  is.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
template<class VoxelType>
void GateInterfileHeader::ReadData(G4String dataFileName, std::vector<VoxelType> & data)
{
  m_dataFileName = dataFileName;
  ReadData(data);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
template<class VoxelType>
void GateInterfileHeader::ReadData(std::vector<VoxelType> & data)
{

  if (!m_isHeaderInfoRead) {
    G4Exception("GateInterfileHeader.cc:ReadData", "NoHeaderInformation", FatalException, "call GateInterfileHeader::ReadHeader first!");
  }

  if (m_dataTypeName == "UNSIGNED INTEGER") {
    if (m_bytePerPixel==1) {
      DoDataRead<unsigned char>(data);
    } else if (m_bytePerPixel==2) {
      DoDataRead<unsigned short>( data);
    } else if (m_bytePerPixel==4) {
      DoDataRead<unsigned int>( data);
    } else if (m_bytePerPixel==8) {
      DoDataRead<unsigned long>( data);
    }
  } else if (m_dataTypeName == "SIGNED INTEGER") {
    if (m_bytePerPixel==1) {
      DoDataRead<char>(data);
    } else if (m_bytePerPixel==2) {
      DoDataRead<short>( data);
    } else if (m_bytePerPixel==4) {
      DoDataRead<int>( data);
    } else if (m_bytePerPixel==8) {
      DoDataRead<long>( data);
    }
  }
  else if (m_dataTypeName == "FLOAT") {
    if (m_bytePerPixel==4) {
      DoDataRead<float>( data);
    } else if (m_bytePerPixel==8) {
      DoDataRead<double>( data);
    }
  }
}


#endif
