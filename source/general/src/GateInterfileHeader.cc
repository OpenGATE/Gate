/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <fstream>

#include "GateInterfileHeader.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"
#include <stdio.h>
#include <string.h>

GateInterfileHeader::GateInterfileHeader()
{

  //m_name = G4String("interfileReader");
  //m_fileName  = G4String("");
  //m_messenger = new GateGeometryVoxelInterfileReaderMessenger(this);

  m_dataFileName = G4String("");
  m_numPlanes=0;
  m_planeThickness=0;
  memset(m_dim,0,sizeof(m_dim));
  memset(m_pixelSize,0,sizeof(m_pixelSize));
  memset(m_matrixSize,0,sizeof(m_matrixSize));
  m_dataTypeName="";
  m_dataByteOrder="BIGENDIAN";
}

//GateInterfileHeader::~GateInterfileHeader()
//{

//}


//void GateInterfileReader::Describe(G4int level)
//{
//  G4cout << " Voxel reader type ---> " << m_name << G4endl;

//  GateVGeometryVoxelReader::Describe(level);
//
//}

void GateInterfileHeader::ReadHeader(std::string & filename)
{

  //m_fileName = fileName;

  FILE* fp=fopen(filename.c_str(),"r");
  if (!fp) {
    G4cerr << G4endl << "Error: Could not open header file '" << filename << "'!" << G4endl;
    return;
  }
  while ( (!feof(fp)) && (!ferror(fp)))
    ReadKey(fp);
  fclose(fp);

  for (G4int i=0; i<2; i++)
    m_matrixSize[i] = m_dim[i] * m_pixelSize[i];

  G4cout << " Header read from       '" << filename << "'" << G4endl;
  G4cout << " Data file name         '" << m_dataFileName << "'" << G4endl;
  G4cout << " Nb of planes:           " << m_numPlanes << G4endl;
  G4cout << " Nb of pixels per plane: " << m_dim[0] << " " << m_dim[1] << G4endl;
  G4cout << " Pixel size:             " << m_pixelSize[0] << " " << m_pixelSize[1] << G4endl;
  G4cout << " Slice thickness:        " << m_planeThickness << G4endl;
  G4cout << " Matrix size:            " << m_matrixSize[0] << " " << m_matrixSize[1] << G4endl;
  G4cout << " Data type:              " << m_dataTypeName << G4endl;
  G4cout << " Data byte order:	 " << m_dataByteOrder << G4endl;
  G4cout << G4endl;

  if ( ( m_dim[0]==0) || ( m_dim[1]==0) || ( m_numPlanes==0) ) {
    G4cerr << G4endl <<"Error: one of the matrix dimensions is zero!" << G4endl;
  return;
  }
}

void GateInterfileHeader::ReadData(std::string filename, std::vector<float> & data)
{
  G4int pixelNumber = m_dim[0]*m_dim[1]*m_numPlanes ;
  /*Set ( pixelNumber );*/

  FILE* fp=fopen(filename.c_str(),"r");
  while ( (!feof(fp)) && (!ferror(fp)))
    ReadKey(fp);
  fclose(fp);

  int l = filename.length();
  filename.replace(l-3,3,"i33");
  //G4cout << filename << G4endl;

  std::ifstream is;
  OpenFileInput(filename, is);

//  FILE* fpp=fopen(m_dataFileName.c_str(),"r");
//  if (!fpp) {
//    G4cerr << G4endl << "Error: Could not open image file '" << m_dataFileName << "'!" << G4endl;
//    return;
//  }

//  G4cout << m_dataTypeName << G4endl;
//
//  if (m_dataTypeName == "UNSIGNED INTEGER") {

//    G4short  *buffer = (G4short*) malloc ( pixelNumber*sizeof(G4short) );
//
//    if (!buffer) {
//      G4cerr << G4endl << "Error: Could not allocate the buffer!" << G4endl;
//      return;
//    }
//
//    G4cout << "je lis l'Interfile U16" << G4endl;
//    G4int NbData = fread( buffer, sizeof(G4short), pixelNumber, fpp);
//    fclose(fpp);
//
//    if ( NbData != pixelNumber ) {
//      G4cerr << G4endl <<"Error: the number of pixels that were read from the data file (" << NbData << ") " << G4endl
//      	      	   << "is inferior to the number computed from its header file (" << pixelNumber << ")!" << G4endl;
//      return;
//    }
//     // Section added by Henri Der Sarkissian : Process the data buffer to match image byte order
//// If the image byte order is different from the native CPU byte order, swap the bytes in buffer
//    if (((BYTE_ORDER == LITTLE_ENDIAN) && (m_dataByteOrder.compare("BIGENDIAN") == 0)) || ((BYTE_ORDER == BIG_ENDIAN) && (m_dataByteOrder.compare("LITTLEENDIAN") == 0))) {
//		SwapBytes(buffer, pixelNumber);
//    }
//  }

  if (m_dataTypeName == "UNSIGNED INTEGER") {

    //G4cout << "datatype=UNSIGNED INTEGER" << G4endl;
    typedef unsigned short VoxelType;
    std::vector<VoxelType> temp(pixelNumber);
    data.resize(pixelNumber);
    is.read((char*)(&(temp[0])), pixelNumber*sizeof(VoxelType));
    for(unsigned int i=0; i<temp.size(); i++) {
      data[i] = (PixelType)temp[i];
      //G4cout << data[i] << G4endl;
    }
  }
  else if (m_dataTypeName == "FLOAT") {

    //G4cout << "datatype=FLOAT" << G4endl;
    typedef float VoxelType;
    std::vector<VoxelType> temp(pixelNumber);
    data.resize(pixelNumber);
    is.read((char*)(&(temp[0])), pixelNumber*sizeof(VoxelType));
    for(unsigned int i=0; i<temp.size(); i++) {
      data[i] = (PixelType)temp[i];
      //G4cout << data[i] << G4endl;
  }
//    G4short  *buffer = (G4short*) malloc ( pixelNumber*sizeof(float) );
//
//    if (!buffer) {
//      G4cerr << G4endl << "Error: Could not allocate the buffer!" << G4endl;
//      return;
//    }
//
//    G4cout << "je lis l'Interfile FLOAT" << G4endl;
//    G4int NbData = fread( buffer, sizeof(float), pixelNumber, fpp);
//    fclose(fpp);
//
//    if ( NbData != pixelNumber ) {
//      G4cerr << G4endl <<"Error: the number of pixels that were read from the data file (" << NbData << ") " << G4endl
//      	      	   << "is inferior to the number computed from its header file (" << pixelNumber << ")!" << G4endl;
//      return;
//    }
    // Section added by Henri Der Sarkissian : Process the data buffer to match image byte order
// If the image byte order is different from the native CPU byte order, swap the bytes in buffer
//    if (((BYTE_ORDER == LITTLE_ENDIAN) && (m_dataByteOrder.compare("BIGENDIAN") == 0)) || ((BYTE_ORDER == BIG_ENDIAN) && (m_dataByteOrder.compare("LITTLEENDIAN") == 0))) {
//		SwapBytes(buffer, pixelNumber);
//    }
  }
}

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
    DD("here");
    DD(value);
    m_dataFileName = std::string(value);
    DD(m_dataFileName);
  } else if ( key ==  "number format" ) {
    if ( (strcmp(value,"float")==0) || (strcmp(value,"FLOAT")==0) )
      m_dataTypeName = "FLOAT";
    else if ( (strcmp(value,"unsigned integer")==0) || (strcmp(value,"UNSIGNED INTEGER")==0) )
      m_dataTypeName = "UNSIGNED INTEGER";
    else
      G4cout << "Unrecognised type name '" << value << "'" << G4endl;
  } else if (key == "imagedata byte order") {
  	if ( strcmp(value,"BIGENDIAN") == 0 )
  		m_dataByteOrder = "BIGENDIAN";
  	else if ( strcmp(value,"LITTLEENDIAN") == 0)
  		m_dataByteOrder = "LITTLEENDIAN";
  	else
  		G4cerr << "Unrecognized data byte order '" + G4String(value) + "', assuming default BIGENDIAN\n" << G4endl;
  } else {
    // G4cout << "Key not processed: '" << key << "'" << G4endl;
  }

}
