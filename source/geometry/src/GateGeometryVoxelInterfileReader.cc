/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include <fstream>
/* PY Descourt 08/09/2009 */
#include "GateRTPhantom.hh"
#include "GateRTPhantomMgr.hh"
/* PY Descourt 08/09/2009 */
#include "GateGeometryVoxelInterfileReader.hh"
#include "GateGeometryVoxelInterfileReaderMessenger.hh"
#include "GateVGeometryVoxelTranslator.hh"
#include "GateMaterialDatabase.hh"
#include "GateVVolume.hh"

#include <stdio.h>
#include <string.h>

GateGeometryVoxelInterfileReader::GateGeometryVoxelInterfileReader(GateVVolume* inserter)
  : GateVGeometryVoxelReader(inserter)
{
    IsFirstFrame = true;
  m_name = G4String("interfileReader");
  m_fileName  = G4String("");
  m_messenger = new GateGeometryVoxelInterfileReaderMessenger(this);

  m_dataFileName = G4String("");
  m_numPlanes=0;
  m_planeThickness=0;
  memset(m_dim,0,sizeof(m_dim));
  memset(m_pixelSize,0,sizeof(m_pixelSize));
  memset(m_matrixSize,0,sizeof(m_matrixSize));
  m_dataTypeName="";
  m_dataByteOrder="BIGENDIAN";
}

GateGeometryVoxelInterfileReader::~GateGeometryVoxelInterfileReader()
{
  if (m_messenger) {
    delete m_messenger;
  }
}


void GateGeometryVoxelInterfileReader::Describe(G4int level)
{
  G4cout << " Voxel reader type ---> " << m_name << G4endl;

  GateVGeometryVoxelReader::Describe(level);

}

void GateGeometryVoxelInterfileReader::ReadFile(G4String fileName)
{

m_fileName = fileName;

  FILE* fp=fopen(m_fileName.c_str(),"r");
  if (!fp) {
    G4cerr << G4endl << "Error: Could not open header file '" << m_fileName << "'!" << G4endl;
    return;
  }
  while ( (!feof(fp)) && (!ferror(fp)))
    ReadKey(fp);
  fclose(fp);

  for (G4int i=0; i<2; i++)
    m_matrixSize[i] = m_dim[i] * m_pixelSize[i];

  // extract folder for fileName if any
  //size_t found;
  //found=fileName.find_last_of("/\\");
  //G4String folder = fileName.substr(0,found);
  //m_dataFileName = folder+"/"+m_dataFileName;
  
  //int l = fileName.length();
  //G4cout << "fileName=" << fileName << G4endl; 
  m_dataFileName=fileName.replace(fileName.length()-3,3,"i33");
  G4cout << "m_dataFileName=" << m_dataFileName << G4endl;

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
    G4cerr << G4endl << "Error: Could not open header file '" << m_dataFileName << "'!" << G4endl;
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

// Section added by Henri Der Sarkissian : Process the data buffer to match image byte order
// If the image byte order is different from the native CPU byte order, swap the bytes in buffer
	if (((BYTE_ORDER == LITTLE_ENDIAN) && (m_dataByteOrder.compare("BIGENDIAN") == 0)) || ((BYTE_ORDER == BIG_ENDIAN) && (m_dataByteOrder.compare("LITTLEENDIAN") == 0))) {
		SwapBytes(buffer, pixelNumber);
	}
// End
  EmptyStore();

  G4String materialName;
  G4int    imageValue;
  G4double dx, dy, dz;
  G4int    nx, ny, nz;

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;

  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;

  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );


  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
	imageValue = buffer[ix+nx*iy+nx*ny*iz];
	materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
	if ( materialName != G4String("NULL") ) {
	  G4Material* material = mMaterialDatabase.GetMaterial(materialName);
	  AddVoxel(ix, iy, iz, material);
	} else {
	  G4cout << "GateGeometryVoxelInterfileReader::ReadFile: WARNING: voxel not added (material translation not found); value: "<< imageValue << G4endl;
	}
      }
    }
  }

 free(buffer);
  UpdateParameters();

  // saves the voxel info through the OutputMgr
  Dump();

  if (m_compressor) {
    Compress();
    EmptyStore();
    G4cout << "GateSourceVoxelInterfileReader::ReadFile: For your information, the voxel store has been emptied." << G4endl;
  }

}


void GateGeometryVoxelInterfileReader::ReadKey(FILE* fp)
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



/*PY Descourt 08/09/2009 */

void GateGeometryVoxelInterfileReader::ReadRTFile(G4String Header_fileName, G4String m_dataFileName)
{

// Check there is a GatePhantom attached to this source

GateRTPhantom *Ph = GateRTPhantomMgr::GetInstance()->CheckGeometryAttached( GetCreator()->GetObjectName() );

if ( Ph != 0)
{G4cout << " The Object "<< Ph->GetName()
<<" is attached to the "<<m_name<<" Geometry Voxel Reader."<<G4endl;
} else { G4cout << " GateGeometryVoxelInterfileReader::ReadFile   WARNING The Object "<< Ph->GetName()
<<" is not attached to any Geometry Voxel Reader."<<G4endl; }



m_fileName = Header_fileName;


  G4double dx, dy, dz;
  G4int    nx, ny, nz;


  if ( IsFirstFrame == true )
   {
    FILE* fp=fopen(m_fileName.c_str(),"r");
    if (!fp) {
              G4cerr << G4endl << "Error: Could not open header file '" << m_fileName << "'!" << G4endl;
              return;
             }
    IsFirstFrame = false;
    while ( (!feof(fp)) && (!ferror(fp)))
    ReadKeyFrame(fp);
    fclose(fp);

    EmptyStore();

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;
  for (G4int i=0; i<2; i++)
    m_matrixSize[i] = m_dim[i] * m_pixelSize[i];


  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;

  SetVoxelNx( nx );
  SetVoxelNy( ny );
  SetVoxelNz( nz );

  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;

  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );




   }

  nx = m_dim[0];
  ny = m_dim[1];
  nz = m_numPlanes;
  dx = m_pixelSize[0];
  dy = m_pixelSize[1];
  dz = m_planeThickness;


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
    G4cerr << G4endl << "Error: Could not open Voxels Data file '" << m_dataFileName << "'!" << G4endl;
    return;
  }


//  G4short  *buffer = (G4short*) malloc ( pixelNumber*sizeof(G4short) );

G4short  *buffer = new G4short[ pixelNumber ];

memset( buffer,0,pixelNumber);


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

// Section added by Henri Der Sarkissian : Process the data buffer to match image byte order
// If the image byte order is different from the native CPU byte order, swap the bytes in buffer
	if (((BYTE_ORDER == LITTLE_ENDIAN) && (m_dataByteOrder.compare("BIGENDIAN") == 0)) || ((BYTE_ORDER == BIG_ENDIAN) && (m_dataByteOrder.compare("LITTLEENDIAN") == 0))) {
		SwapBytes(buffer, pixelNumber);
	}

// End

//  EmptyStore();

  G4String materialName;
  G4int    imageValue;

  G4cout << "nx ny nz: " << nx << " " << ny << " " << nz << G4endl;
  G4cout << "dx dy dz: " << dx << " " << dy << " " << dz << G4endl;

//  SetVoxelSize( G4ThreeVector(dx, dy, dz) * mm );


// G4cout << " GateGeometryVoxelInterfileReader::ReadFile Vowel Size Vector  "<< G4ThreeVector(dx, dy, dz) * mm << G4endl;


  for (G4int iz=0; iz<nz; iz++) {
    for (G4int iy=0; iy<ny; iy++) {
      for (G4int ix=0; ix<nx; ix++) {
 imageValue = buffer[ix+nx*iy+nx*ny*iz];
 materialName = m_voxelTranslator->TranslateToMaterial(imageValue);
     if ( materialName != G4String("NULL") ) {
   G4Material* material = mMaterialDatabase.GetMaterial(materialName);
   AddVoxel(ix, iy, iz, material);
 } else {
   G4cout << "GateGeometryVoxelInterfileReader::ReadFile: WARNING: voxel not added (material translation not found); value: "<< imageValue << G4endl;
 }
      }
    }
  }

delete [] buffer;

  if (m_compressor) {

    m_compressor->Initialize();
    Compress();

    G4cout << "---------- Gate Voxels Compressor Statistics ---------"<<G4endl;
    G4cout << "  Initial number of voxels in The Phantom      : " << GetNumberOfVoxels() << G4endl;
    G4cout << "  number of compressed voxels                  : " << m_compressor->GetNbOfCopies() << G4endl;
    G4cout << "  Compression achieved                                            : " << m_compressor->GetCompressionRatio() << " %"  << G4endl;
    G4cout << "-------------------------------------------------------------------"<<G4endl;



//    EmptyStore();
//    G4cout << "GateGeometryVoxelInterfileReader::ReadFile: For your information, the voxel store has been emptied." << G4endl;

  }

}

void GateGeometryVoxelInterfileReader::ReadKeyFrame(FILE* fp)
{
  if ( (feof(fp)) && (ferror(fp)))
    return ;


  char keyBuffer[256],valueBuffer[256];
  if( fscanf(fp,"%[^=]",keyBuffer) == EOF )
	{
		G4cerr << "Number of receiving arguments failed!!!" << G4endl;
	}
  if( fscanf(fp,"%[^\n]",valueBuffer) == EOF )
	{
		G4cerr << "Number of receiving arguments failed!!!" << G4endl;
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

//G4cout << " key is " << key <<G4endl;

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

/*PY Descourt 08/09/2009 */
