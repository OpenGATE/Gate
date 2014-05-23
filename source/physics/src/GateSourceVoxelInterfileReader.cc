/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "G4SystemOfUnits.hh"
#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
#include "GateSourceVoxelInterfileReader.hh"
#include "GateSourceVoxelInterfileReaderMessenger.hh"
#include "GateVSourceVoxelTranslator.hh"
#include "GateMessageManager.hh"
#include <fstream>
#include <stdio.h>
#include <string.h>

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateSourceVoxelInterfileReader::GateSourceVoxelInterfileReader(GateVSource* source)
  : GateVSourceVoxelReader(source)
{
  nVerboseLevel = 0;

  m_name = G4String("interfileReader");

  m_messenger = new GateSourceVoxelInterfileReaderMessenger(this);
  
  m_fileName  = G4String("");

  m_dataFileName = G4String("");
  m_numPlanes=0;
  m_planeThickness=0;
  memset(m_dim,0,sizeof(m_dim));
  memset(m_pixelSize,0,sizeof(m_pixelSize));
  memset(m_matrixSize,0,sizeof(m_matrixSize));
  m_dataTypeName="";
  m_dataByteOrder="BIGENDIAN";
}

GateSourceVoxelInterfileReader::~GateSourceVoxelInterfileReader()
{
  delete m_messenger;
}

void GateSourceVoxelInterfileReader::ReadFile(G4String fileName)
{
  if (!m_voxelTranslator) {
    G4cout << "GateSourceVoxelImageReader::ReadFile: ERROR : insert a translator first" << G4endl;
    return;
  }
G4cout << "GateSourceVoxelImageReader::ReadFile : fileName: " << m_fileName << G4endl;
  
  m_fileName = fileName;

  FILE* fp=fopen(m_fileName.c_str(),"r");
  if (!fp) {
    G4cerr << G4endl << "Error: Could not open header file '" << m_fileName << "'!" << G4endl;
    return;
  }



  while ( (!feof(fp)) && (!ferror(fp)))
    ReadKey(fp);
  fclose(fp);

  // extract folder for fileName if any
  //size_t found;
  //found=fileName.find_last_of("/\\");
  //G4String folder = fileName.substr(0,found);
  //m_dataFileName = folder+"/"+m_dataFileName;
  m_dataFileName=fileName.replace(fileName.length()-3,3,"i33");
    
  for (G4int i=0; i<2; i++)
    m_matrixSize[i] = m_dim[i] * m_pixelSize[i];

  G4cout << " Header read from       '" << m_fileName << "'" << G4endl;
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
  G4int pixelNumber = m_dim[0]*m_dim[1]*m_numPlanes ;
  /*Set ( pixelNumber );*/

  FILE* fpp=fopen(m_dataFileName.c_str(),"r");
  if (!fpp) {
    G4cerr << G4endl << "Error: Could not open data file '" << m_dataFileName << "'!" << G4endl;
    return;
  }

  G4short  *buffer = (G4short*) malloc ( pixelNumber*sizeof(G4short) );
  if (!buffer) {
    G4cerr << G4endl << "Error: Could not allocate the buffer!" << G4endl;
    return;
  }

  G4int NbData = fread( buffer, sizeof(G4short), pixelNumber, fpp);
  fclose(fpp);
  if ( NbData != pixelNumber ) {
      G4cerr << G4endl <<"Error: the number of pixels that were read from the data file (" << NbData << ") " << G4endl
      	      	   << "is inferior to the number computed from its header file (" << pixelNumber << ")!" << G4endl;
      return;
  }

// Added by Henri Der Sarkissian : Process the data buffer to match image byte order
// If the image byte order is different from the native CPU byte order, swap the bytes in buffer
	if (((BYTE_ORDER == LITTLE_ENDIAN) && (m_dataByteOrder.compare("BIGENDIAN") == 0)) || ((BYTE_ORDER == BIG_ENDIAN) && (m_dataByteOrder.compare("LITTLEENDIAN") == 0))) {
		SwapBytes(buffer, pixelNumber);
	}
// End

  G4double activity;
  G4int    imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
	imageValue = buffer[ix+nx*iy+nx*ny*iz];
	activity = m_voxelTranslator->TranslateToActivity(imageValue);

         if (activity > 0.) {
	  AddVoxel(ix, iy, iz, activity);
	}
      }
    }
  }
  free(buffer);

  PrepareIntegratedActivityMap();

}

void GateSourceVoxelInterfileReader::ReadKey(FILE* fp)
{
  if ( (feof(fp)) && (ferror(fp)))
    return ;


  char keyBuffer[256],valueBuffer[256];
  if( fscanf(fp,"%[^=]",keyBuffer) == EOF )
	{
		;//G4cerr << "Number of receiving arguments failed!!!" << G4endl;
	}
  if( fscanf(fp,"%[^\n]",valueBuffer) == EOF )
	{
		;//G4cerr << "Number of receiving arguments failed!!!" << G4endl;
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


/* PY Descourt 08/09/2009 */

void GateSourceVoxelInterfileReader::ReadRTFile(G4String header_fileName, G4String dataFileName)
{


Initialize();

m_dataFileName = dataFileName;

// Check there is a GatePhantom attached

GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckSourceAttached( m_name );



if ( Ph != 0)
{G4cout << " The Object "<< Ph->GetName()
<<" is attached to the "<<m_name<<" Source Voxel Reader." << G4endl;
}  else { G4cout << " GateSourceVoxelInterfileReader::ReadFile   WARNING The Object "<< Ph->GetName()
<<" is not attached to any Geometry Voxel Reader."<<G4endl; }

if (!m_voxelTranslator) {
    G4cout << "GateSourceVoxelInterfileReader::ReadFile: ERROR : insert a translator first" << G4endl;
    return;
  }

  if ( IsFirstFrame == true )
 {

  FILE* fp=fopen(header_fileName.c_str(),"r");
  if (!fp) {
    G4cerr << G4endl << "Error: Could not open header file '" << header_fileName << "'!" << G4endl;
    return;
  }

  IsFirstFrame = false;
  while ( (!feof(fp)) && (!ferror(fp)))
  ReadKeyFrames(fp);
  fclose(fp);

 }


  for (G4int i=0; i<2; i++)
    m_matrixSize[i] = m_dim[i] * m_pixelSize[i];

  G4cout << " Header read from       '" << header_fileName << "'" << G4endl;
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
  G4int pixelNumber = m_dim[0]*m_dim[1]*m_numPlanes ;
  /*Set ( pixelNumber );*/

  FILE* fpp=fopen(m_dataFileName.c_str(),"r");
  if (!fpp) {
    G4cerr << G4endl << "Error: Could not open header file '" << m_dataFileName << "'!" << G4endl;
    return;
  }


  G4short  *buffer = new G4short[ pixelNumber ];

  if (!buffer) {
    G4cerr << G4endl << "Error: Could not allocate the buffer!" << G4endl;
    return;
  }

  G4int NbData = fread( buffer, sizeof(G4short), pixelNumber, fpp);
  fclose(fpp);
  if ( NbData != pixelNumber ) {
      G4cerr << G4endl <<"Error: the number of pixels that were read from the data file (" << NbData << ") " << G4endl
                 << "is inferior to the number computed from its header file (" << pixelNumber << ")!" << G4endl;
      return;
  }

// Added by Henri Der Sarkissian : Process the data buffer to match image byte order
// If the image byte order is different from the native CPU byte order, swap the bytes in buffer
	if (((BYTE_ORDER == LITTLE_ENDIAN) && (m_dataByteOrder.compare("BIGENDIAN") == 0)) || ((BYTE_ORDER == BIG_ENDIAN) && (m_dataByteOrder.compare("LITTLEENDIAN") == 0))) {
		for (G4int i=0; i < pixelNumber; ++i) {
			SwapBytes(buffer, pixelNumber);
		}
	}
// End

  G4double activity;
  G4int    imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );

  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
 imageValue = buffer[ix+nx*iy+nx*ny*iz];
 activity = m_voxelTranslator->TranslateToActivity(imageValue);

         if (activity > 0.) AddVoxel_FAST(ix, iy, iz, activity);

     }
    }
  }

delete [] buffer;

  PrepareIntegratedActivityMap();

}

void GateSourceVoxelInterfileReader::ReadKeyFrames(FILE* fp)
{
  if ( (feof(fp)) && (ferror(fp)))
    return ;


  char keyBuffer[256],valueBuffer[256];
  if( fscanf(fp,"%[^=]",keyBuffer) == EOF )
	{
		;//G4cerr << "Number of receiving arguments failed!!!" << G4endl;
	}
  if( fscanf(fp,"%[^\n]",valueBuffer) == EOF )
	{
		;//G4cerr << "Number of receiving arguments failed!!!" << G4endl;
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
/* PY Descourt 08/09/2009 */
