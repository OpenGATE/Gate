/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include <fstream>
#include <stdio.h>
#include <string.h>

#include "GateInterfileHeader.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"

//-----------------------------------------------------------------------------
GateInterfileHeader::GateInterfileHeader()
{
  m_headerFileName = G4String("");
  m_dataFileName = G4String("");
  m_numPlanes=0;
  m_planeThickness=0;
  memset(m_dim,0,sizeof(m_dim));
  memset(m_pixelSize,0,sizeof(m_pixelSize));
  memset(m_matrixSize,0,sizeof(m_matrixSize));
  m_dataTypeName="";
  m_dataByteOrder= BIG_ENDIAN;
  m_bytePerPixel = 2;
  m_offset = 0;
  m_isHeaderInfoRead = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateInterfileHeader::ReadHeader(G4String headerFileName)
{
  m_headerFileName = headerFileName;
  FILE* fp=fopen(m_headerFileName.c_str(),"r");
  if (!fp) {
    G4cerr << G4endl << "Error: Could not open header file '" << m_headerFileName << "'!" << G4endl;
    return;
  }
  while ( (!feof(fp)) && (!ferror(fp)))
    ReadKey(fp);
  fclose(fp);

  if (m_dataFileName.empty()) {
	  // guess filename
	  m_dataFileName = headerFileName.replace(headerFileName.length()-3, 3, "i33");
  }

  for (G4int i=0; i<2; i++)
    m_matrixSize[i] = m_dim[i] * m_pixelSize[i];

  G4cout << " Header read from       '" << m_headerFileName << "'" << G4endl;
  G4cout << " Data file name         '" << m_dataFileName << "'" << G4endl;
  G4cout << " Nb of planes:           " << m_numPlanes << G4endl;
  G4cout << " Nb of pixels per plane: " << m_dim[0] << " " << m_dim[1] << G4endl;
  G4cout << " Pixel size:             " << m_pixelSize[0] << " " << m_pixelSize[1] << G4endl;
  G4cout << " Slice thickness:        " << m_planeThickness << G4endl;
  G4cout << " Matrix size:            " << m_matrixSize[0] << " " << m_matrixSize[1] << G4endl;
  G4cout << " Data type:              " << m_dataTypeName << G4endl;
  G4cout << " Bytes per pixel:        " << m_bytePerPixel << G4endl;
  G4cout << " Data byte order:        " << (m_dataByteOrder==LITTLE_ENDIAN ? "LITTLEENDIAN" : "BIGENDIAN") << G4endl;
  G4cout << " Machine byte order:     " << (BYTE_ORDER==LITTLE_ENDIAN ? "LITTLEENDIAN" : "BIGENDIAN") << G4endl;
  G4cout << G4endl;

  if ( ( m_dim[0]==0) || ( m_dim[1]==0) || ( m_numPlanes==0) ) {
    G4cerr << G4endl <<"Error: one of the matrix dimensions is zero!" << G4endl;
    return;
  }
  m_isHeaderInfoRead = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateInterfileHeader::ReadData(G4String dataFileName, std::vector<PixelType> & data)
{
  m_dataFileName = dataFileName;
  ReadData(data);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateInterfileHeader::ReadData(std::vector<PixelType> & data)
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
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
template <class VoxelType> void GateInterfileHeader::DoDataRead(std::vector<PixelType> &data) {
  G4int pixelNumber = m_dim[0]*m_dim[1]*m_numPlanes ;
  std::ifstream is;
  OpenFileInput(m_dataFileName, is);
  std::vector<VoxelType> temp(pixelNumber);

  data.resize(pixelNumber);
  is.seekg(m_offset, std::ios::beg);
  is.read((char*)(&(temp[0])), pixelNumber*sizeof(VoxelType));
  if (!is) {
      G4cerr << G4endl <<"Error: the number of pixels that were read from the data file (" << is.gcount() << ") " << G4endl
	  << "is inferior to the number computed from its header file (" << pixelNumber << ")!" << G4endl;
      G4Exception( "GateInterfileHeader.cc InterfileTooShort", "InterfileTooShort", FatalException, "Correct problem then try again... Sorry!" );
  }
  for(unsigned int i=0; i<temp.size(); i++) {
      if ( BYTE_ORDER != m_dataByteOrder ) {
	  VoxelType t = temp[i];
	  GateMachine::SwapEndians( t );
	  temp[i] = t;
//	  GateMachine::SwapEndians( temp[i] );
      }
      data[i] = (PixelType)temp[i];
  }
  is.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateInterfileHeader::ReadKey(FILE* fp)
{
  if ( (feof(fp)) && (ferror(fp)))
    return ;

  char keyBuffer[256],valueBuffer[256];
  if( fscanf(fp,"%[^=]",keyBuffer) == EOF )
    {
      ;
    }
  if( fscanf(fp,"%[^\n]",valueBuffer) == EOF )
    {
      ;
    }

  char *keyPtr = keyBuffer;
  while ( ( *keyPtr == '!' ) || ( *keyPtr == ' ' ) || ( *keyPtr == '\n' ) )
    keyPtr++;

  char *endptr = keyPtr + strlen(keyPtr) - 1;
  *(endptr--)=0;
  while ( *endptr == ' ')
    *(endptr--)=0;
  std::string key(keyPtr);

  char *value = valueBuffer+1;
  while ( *value == ' ')
    value++;

  if ( key ==  "matrix size [1]" ) {
    sscanf(value,"%d",m_dim);
  } else if ( key ==  "matrix size [2]" ) {
    sscanf(value,"%d",m_dim+1);
  } else if ( ( key ==  "number of slices" ) || (key ==  "number of images") ) {
    sscanf(value,"%d",&m_numPlanes);
  } else if ( key ==  "scaling factor (mm/pixel) [1]" ) {
    sscanf(value,"%f",m_pixelSize);
  } else if ( key ==  "scaling factor (mm/pixel) [2]" ) {
    sscanf(value,"%f",m_pixelSize+1);
  } else if ( key ==  "slice thickness (pixels)" ) {
    sscanf(value,"%f",&m_planeThickness);
  } else if ( key ==  "name of data file" ) {
    m_dataFileName = std::string(value);
  } else if ( key ==  "number format" ) {
    if ( (strcmp(value,"float")==0) || (strcmp(value,"FLOAT")==0) )
      m_dataTypeName = "FLOAT";
    else if  ( (strcmp(value,"short float")==0) || (strcmp(value,"SHORT FLOAT")==0) ) {
       m_dataTypeName = "FLOAT";
       m_bytePerPixel = 4;
    } else if ( (strcmp(value,"long float")==0) || (strcmp(value,"LONG FLOAT")==0) ) {
       m_dataTypeName = "FLOAT";
       m_bytePerPixel = 8;
    }
    else if ( (strcmp(value,"unsigned integer")==0) || (strcmp(value,"UNSIGNED INTEGER")==0) )
      m_dataTypeName = "UNSIGNED INTEGER";
    else if ( (strcmp(value,"signed integer")==0) || (strcmp(value,"SIGNED INTEGER")==0) )
      m_dataTypeName = "SIGNED INTEGER";
    else
      G4cout << "Unrecognised type name '" << value << "'" << G4endl;
  } else if (key == "imagedata byte order") {
    if ( strcmp(value,"BIGENDIAN") == 0 )
      m_dataByteOrder = BIG_ENDIAN;
    else if ( strcmp(value,"LITTLEENDIAN") == 0)
      m_dataByteOrder = LITTLE_ENDIAN;
    else
      G4cerr << "Unrecognized data byte order '" + G4String(value) + "', assuming default BIGENDIAN\n" << G4endl;
  } else if ( key ==  "number of bytes per pixel" ) {
      sscanf(value,"%d",&m_bytePerPixel);
  } else if ( key ==  "data offset in bytes" ) {
      sscanf(value,"%d",&m_offset);
  } else {
    // G4cout << "Key not processed: '" << key << "'" << G4endl;
  }
}
//-----------------------------------------------------------------------------
